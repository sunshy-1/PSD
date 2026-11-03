"""Execute one code benchmark with kernel filesystem/syscall restrictions."""
import contextlib
import ctypes
import ctypes.util
import errno
import io
import json
import os
from pathlib import Path
import random
import resource
import sys
import time

class RuleSet(ctypes.Structure):
    _fields_=[('handled_access_fs',ctypes.c_uint64)]

class PathRule(ctypes.Structure):
    _pack_=1
    _fields_=[('allowed_access',ctypes.c_uint64),('parent_fd',ctypes.c_int32)]

def restrict_filesystem(libc,work):
    libc.syscall.restype=ctypes.c_long
    abi=libc.syscall(444,0,0,1)
    if abi<3: raise RuntimeError('Landlock ABI >=3 is required')
    handled=(1<<15)-1
    attributes=RuleSet(handled)
    fd=libc.syscall(444,ctypes.byref(attributes),ctypes.sizeof(attributes),0)
    if fd<0: raise OSError(ctypes.get_errno(),'landlock_create_ruleset')
    read=(1<<2)|(1<<3)
    system_root = Path(Path(sys.executable).anchor)
    paths=[(str(system_root/'usr'),read),(str(system_root/'lib'),read),
           (str(system_root/'lib64'),read),(sys.prefix,read),
           (str(system_root/'etc'/'ld.so.cache'),1<<2),(os.devnull,(1<<1)|(1<<2)),
           (str(work),handled&~(1<<0))]
    for name,allowed in paths:
        if not Path(name).exists(): continue
        pathfd=os.open(name,os.O_PATH|os.O_CLOEXEC)
        rule=PathRule(allowed,pathfd)
        result=libc.syscall(445,fd,1,ctypes.byref(rule),0)
        os.close(pathfd)
        if result<0: raise OSError(ctypes.get_errno(),'landlock_add_rule: '+name)
    if libc.prctl(38,1,0,0,0)!=0: raise OSError(ctypes.get_errno(),'PR_SET_NO_NEW_PRIVS')
    if libc.syscall(446,fd,0)<0: raise OSError(ctypes.get_errno(),'landlock_restrict_self')
    os.close(fd)
    return int(abi)

def restrict_syscalls(library):
    library.seccomp_init.argtypes=[ctypes.c_uint32]
    library.seccomp_init.restype=ctypes.c_void_p
    library.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p]
    library.seccomp_syscall_resolve_name.restype=ctypes.c_int
    library.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
    library.seccomp_load.argtypes=[ctypes.c_void_p]
    library.seccomp_release.argtypes=[ctypes.c_void_p]
    ctx=library.seccomp_init(0x00050000|errno.EPERM)
    if not ctx: raise RuntimeError('seccomp_init failed')
    names=('read write readv writev pread64 pwrite64 close close_range lseek fstat newfstatat stat statx lstat '
           'open openat access faccessat faccessat2 readlink readlinkat getdents getdents64 getcwd chdir '
           'mmap mprotect munmap mremap madvise brk futex futex_waitv '
           'rt_sigaction rt_sigprocmask rt_sigreturn sigaltstack '
           'clock_gettime clock_getres gettimeofday time getrusage getrandom uname sysinfo '
           'getpid getppid gettid getuid geteuid getgid getegid getgroups '
           'sched_yield sched_getaffinity nanosleep clock_nanosleep '
           'fcntl ioctl dup dup2 dup3 pipe pipe2 poll ppoll select pselect6 '
           'mkdir mkdirat unlink unlinkat rmdir rename renameat renameat2 '
           'umask fsync fdatasync ftruncate truncate '
           'exit exit_group').split()
    for name in names:
        number=library.seccomp_syscall_resolve_name(name.encode())
        if number>=0 and library.seccomp_rule_add(ctx,0x7fff0000,number,0)!=0:
            raise RuntimeError('Cannot allow syscall '+name)
    if library.seccomp_load(ctx)!=0: raise RuntimeError('seccomp_load failed')
    library.seccomp_release(ctx)

def main():
    payload=json.loads(sys.stdin.read())
    program=payload['program']
    work=Path.cwd()
    library=ctypes.CDLL(ctypes.util.find_library('seccomp'),use_errno=True)
    libc=ctypes.CDLL(None,use_errno=True)
    resource.setrlimit(resource.RLIMIT_CPU,(3,3))
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE,(1024**2,1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE,(64,64))
    resource.setrlimit(resource.RLIMIT_NPROC,(0,0))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    random.seed(0)
    abi=restrict_filesystem(libc,work)
    restrict_syscalls(library)
    start=time.perf_counter()
    result={'passed':False,'error':None,'landlock_abi':abi,'seccomp':True}
    try:
        with open(os.devnull,'w',encoding='utf-8') as sink, contextlib.redirect_stdout(sink),contextlib.redirect_stderr(sink):
            exec(compile(program,'<candidate>','exec'),{'__name__':'__main__'})
        result['passed']=True
    except BaseException as exc:
        result['error']=type(exc).__name__+': '+str(exc)[:500]
    result['seconds']=time.perf_counter()-start
    os.write(1,(json.dumps(result)+'\n').encode())

if __name__=='__main__': main()
