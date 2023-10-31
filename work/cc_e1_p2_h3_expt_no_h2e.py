import numpy, functools
einsum = functools.partial(numpy.einsum, optimize=True)

def exp_nb(cc_obj=None, amp=None, lam=None):
    """
    Generated by gen-cceqs.py at 2023-10-30 21:55:29
    Machine Info: Darwin 23.0.0
    Hostname:     Junjies-MacBook-Air.local
    Git Branch:   pauling
    Git Commit:   b237a1287fb6f44f6d8fbbb082e54b991b7ebf4b

    xi: b+
    xi: b

    """
    res  =     1.000000 * einsum('I,J->IJ'           , xi, xi)
    return res

def exp_na_vo(cc_obj=None, amp=None, lam=None):
    """
    Generated by gen-cceqs.py at 2023-10-30 21:55:29
    Machine Info: Darwin 23.0.0
    Hostname:     Junjies-MacBook-Air.local
    Git Branch:   pauling
    Git Commit:   b237a1287fb6f44f6d8fbbb082e54b991b7ebf4b

    xi: b+
    xi: b

    """
    
    return res

def exp_na_vo_xb(cc_obj=None, amp=None, lam=None):
    """
    Generated by gen-cceqs.py at 2023-10-30 21:55:29
    Machine Info: Darwin 23.0.0
    Hostname:     Junjies-MacBook-Air.local
    Git Branch:   pauling
    Git Commit:   b237a1287fb6f44f6d8fbbb082e54b991b7ebf4b

    xi: b+
    xi: b

    """
    
    return res

def exp_na_ov(cc_obj=None, amp=None, lam=None):
    """
    Generated by gen-cceqs.py at 2023-10-30 21:55:29
    Machine Info: Darwin 23.0.0
    Hostname:     Junjies-MacBook-Air.local
    Git Branch:   pauling
    Git Commit:   b237a1287fb6f44f6d8fbbb082e54b991b7ebf4b

    xi: b+
    xi: b

    """
    
    return res

def exp_na_ov_xb(cc_obj=None, amp=None, lam=None):
    """
    Generated by gen-cceqs.py at 2023-10-30 21:55:29
    Machine Info: Darwin 23.0.0
    Hostname:     Junjies-MacBook-Air.local
    Git Branch:   pauling
    Git Commit:   b237a1287fb6f44f6d8fbbb082e54b991b7ebf4b

    xi: b+
    xi: b

    """
    
    return res

def exp_na_oo(cc_obj=None, amp=None, lam=None):
    """
    Generated by gen-cceqs.py at 2023-10-30 21:55:29
    Machine Info: Darwin 23.0.0
    Hostname:     Junjies-MacBook-Air.local
    Git Branch:   pauling
    Git Commit:   b237a1287fb6f44f6d8fbbb082e54b991b7ebf4b

    xi: b+
    xi: b

    """
    res  =     1.000000 * einsum('ji->ij'            , cc_obj.delta.oo)
    return res

def exp_na_oo_xb(cc_obj=None, amp=None, lam=None):
    """
    Generated by gen-cceqs.py at 2023-10-30 21:55:29
    Machine Info: Darwin 23.0.0
    Hostname:     Junjies-MacBook-Air.local
    Git Branch:   pauling
    Git Commit:   b237a1287fb6f44f6d8fbbb082e54b991b7ebf4b

    xi: b+
    xi: b

    """
    res  =     2.000000 * einsum('I,ji->ijI'         , xi, cc_obj.delta.oo)
    return res

def exp_na_vv(cc_obj=None, amp=None, lam=None):
    """
    Generated by gen-cceqs.py at 2023-10-30 21:55:29
    Machine Info: Darwin 23.0.0
    Hostname:     Junjies-MacBook-Air.local
    Git Branch:   pauling
    Git Commit:   b237a1287fb6f44f6d8fbbb082e54b991b7ebf4b

    xi: b+
    xi: b

    """
    
    return res

def exp_na_vv_xb(cc_obj=None, amp=None, lam=None):
    """
    Generated by gen-cceqs.py at 2023-10-30 21:55:29
    Machine Info: Darwin 23.0.0
    Hostname:     Junjies-MacBook-Air.local
    Git Branch:   pauling
    Git Commit:   b237a1287fb6f44f6d8fbbb082e54b991b7ebf4b

    xi: b+
    xi: b

    """
    
    return res
