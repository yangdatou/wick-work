import numpy, functools
einsum = functools.partial(numpy.einsum, optimize=True)

def exp_nb(cc_obj=None, amp=None, lam=None):
    """
    Generated by gen-cceqs.py at 2023-10-30 22:53:31
    Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    Hostname:     pauling029
    Git Branch:   pauling
    Git Commit:   4e9ab570d6d42b6089a0fa9d0fb3f6934e6e07ba

    amp[0]: a+_v a_o
    cc_obj.xi: b+
    cc_obj.xi: b
    amp[1]: b+
    amp[2]: b+ a+_v a_o
    lam[0]: a+_o a_v
    lam[1]: b
    lam[2]: b a+_o a_v

    """
    res  =     1.000000 * einsum('I,J->IJ'           , lam[1], amp[1])
    res +=     1.000000 * einsum('I,J->IJ'           , lam[1], cc_obj.xi)
    res +=     1.000000 * einsum('Iia,Jai->IJ'       , lam[2], amp[2])
    return res

def exp_na_vo(cc_obj=None, amp=None, lam=None):
    """
    Generated by gen-cceqs.py at 2023-10-30 22:53:31
    Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    Hostname:     pauling029
    Git Branch:   pauling
    Git Commit:   4e9ab570d6d42b6089a0fa9d0fb3f6934e6e07ba

    amp[0]: a+_v a_o
    cc_obj.xi: b+
    cc_obj.xi: b
    amp[1]: b+
    amp[2]: b+ a+_v a_o
    lam[0]: a+_o a_v
    lam[1]: b
    lam[2]: b a+_o a_v

    """
    res  =     1.000000 * einsum('ia->ia'            , lam[0])
    return res

def exp_na_vo_xb(cc_obj=None, amp=None, lam=None):
    """
    Generated by gen-cceqs.py at 2023-10-30 22:53:31
    Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    Hostname:     pauling029
    Git Branch:   pauling
    Git Commit:   4e9ab570d6d42b6089a0fa9d0fb3f6934e6e07ba

    amp[0]: a+_v a_o
    cc_obj.xi: b+
    cc_obj.xi: b
    amp[1]: b+
    amp[2]: b+ a+_v a_o
    lam[0]: a+_o a_v
    lam[1]: b
    lam[2]: b a+_o a_v

    """
    res  =     1.000000 * einsum('Iia->iaI'          , lam[2])
    res +=     1.000000 * einsum('ia,I->iaI'         , lam[0], amp[1])
    res +=     2.000000 * einsum('ia,I->iaI'         , lam[0], cc_obj.xi)
    return res

def exp_na_ov(cc_obj=None, amp=None, lam=None):
    """
    Generated by gen-cceqs.py at 2023-10-30 22:53:31
    Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    Hostname:     pauling029
    Git Branch:   pauling
    Git Commit:   4e9ab570d6d42b6089a0fa9d0fb3f6934e6e07ba

    amp[0]: a+_v a_o
    cc_obj.xi: b+
    cc_obj.xi: b
    amp[1]: b+
    amp[2]: b+ a+_v a_o
    lam[0]: a+_o a_v
    lam[1]: b
    lam[2]: b a+_o a_v

    """
    res  =     1.000000 * einsum('ai->ai'            , amp[0])
    res +=     1.000000 * einsum('I,Iai->ai'         , lam[1], amp[2])
    return res

def exp_na_ov_xb(cc_obj=None, amp=None, lam=None):
    """
    Generated by gen-cceqs.py at 2023-10-30 22:53:31
    Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    Hostname:     pauling029
    Git Branch:   pauling
    Git Commit:   4e9ab570d6d42b6089a0fa9d0fb3f6934e6e07ba

    amp[0]: a+_v a_o
    cc_obj.xi: b+
    cc_obj.xi: b
    amp[1]: b+
    amp[2]: b+ a+_v a_o
    lam[0]: a+_o a_v
    lam[1]: b
    lam[2]: b a+_o a_v

    """
    res  =     1.000000 * einsum('Iai->aiI'          , amp[2])
    res +=     1.000000 * einsum('I,ai->aiI'         , lam[1], amp[0])
    return res

def exp_na_oo(cc_obj=None, amp=None, lam=None):
    """
    Generated by gen-cceqs.py at 2023-10-30 22:53:31
    Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    Hostname:     pauling029
    Git Branch:   pauling
    Git Commit:   4e9ab570d6d42b6089a0fa9d0fb3f6934e6e07ba

    amp[0]: a+_v a_o
    cc_obj.xi: b+
    cc_obj.xi: b
    amp[1]: b+
    amp[2]: b+ a+_v a_o
    lam[0]: a+_o a_v
    lam[1]: b
    lam[2]: b a+_o a_v

    """
    res  =     1.000000 * einsum('ji->ij'            , cc_obj.delta.oo)
    res +=    -1.000000 * einsum('ia,aj->ij'         , lam[0], amp[0])
    res +=    -1.000000 * einsum('Iia,Iaj->ij'       , lam[2], amp[2])
    return res

def exp_na_oo_xb(cc_obj=None, amp=None, lam=None):
    """
    Generated by gen-cceqs.py at 2023-10-30 22:53:31
    Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    Hostname:     pauling029
    Git Branch:   pauling
    Git Commit:   4e9ab570d6d42b6089a0fa9d0fb3f6934e6e07ba

    amp[0]: a+_v a_o
    cc_obj.xi: b+
    cc_obj.xi: b
    amp[1]: b+
    amp[2]: b+ a+_v a_o
    lam[0]: a+_o a_v
    lam[1]: b
    lam[2]: b a+_o a_v

    """
    res  =     1.000000 * einsum('I,ji->ijI'         , lam[1], cc_obj.delta.oo)
    res +=     1.000000 * einsum('I,ji->ijI'         , amp[1], cc_obj.delta.oo)
    res +=     2.000000 * einsum('I,ji->ijI'         , cc_obj.xi, cc_obj.delta.oo)
    res +=    -1.000000 * einsum('ia,Iaj->ijI'       , lam[0], amp[2])
    res +=    -1.000000 * einsum('Iia,aj->ijI'       , lam[2], amp[0])
    res +=     1.000000 * einsum('ia,Iai,kj->jkI'    , lam[0], amp[2], cc_obj.delta.oo)
    return res

def exp_na_vv(cc_obj=None, amp=None, lam=None):
    """
    Generated by gen-cceqs.py at 2023-10-30 22:53:31
    Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    Hostname:     pauling029
    Git Branch:   pauling
    Git Commit:   4e9ab570d6d42b6089a0fa9d0fb3f6934e6e07ba

    amp[0]: a+_v a_o
    cc_obj.xi: b+
    cc_obj.xi: b
    amp[1]: b+
    amp[2]: b+ a+_v a_o
    lam[0]: a+_o a_v
    lam[1]: b
    lam[2]: b a+_o a_v

    """
    res  =     1.000000 * einsum('ia,bi->ba'         , lam[0], amp[0])
    res +=     1.000000 * einsum('Iia,Ibi->ba'       , lam[2], amp[2])
    return res

def exp_na_vv_xb(cc_obj=None, amp=None, lam=None):
    """
    Generated by gen-cceqs.py at 2023-10-30 22:53:31
    Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    Hostname:     pauling029
    Git Branch:   pauling
    Git Commit:   4e9ab570d6d42b6089a0fa9d0fb3f6934e6e07ba

    amp[0]: a+_v a_o
    cc_obj.xi: b+
    cc_obj.xi: b
    amp[1]: b+
    amp[2]: b+ a+_v a_o
    lam[0]: a+_o a_v
    lam[1]: b
    lam[2]: b a+_o a_v

    """
    res  =     1.000000 * einsum('ia,Ibi->baI'       , lam[0], amp[2])
    res +=     1.000000 * einsum('Iia,bi->baI'       , lam[2], amp[0])
    return res
