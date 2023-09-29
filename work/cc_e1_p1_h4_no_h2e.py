import numpy, functools
einsum = functools.partial(numpy.einsum, optimize=True)

def res_0(cc_obj, amp):
    """
    Generated by gen-cceqs.py at 2023-09-20 13:45:00
    Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    Hostname:     pauling001
    Git Branch:   pauling
    Git Commit:   f22052967d28d07b3f2de001f71f7a160db878c0

    amp[0]: a+_v a_o
    amp[1]: b+
    amp[2]: b+ a+_v a_o
    res   :  a+_o a_v

    ih = 0, ibra = 0, len(tmp.terms) = 1
    ih = 1, ibra = 0, len(tmp.terms) = 6
    ih = 2, ibra = 0, len(tmp.terms) = 6
    ih = 3, ibra = 0, len(tmp.terms) = 1
    ih = 4, ibra = 0, len(tmp.terms) = 0
    """
    res  =     1.000000 * einsum('ai->ai'            , cc_obj.h1e.vo)
    res +=     1.000000 * einsum('I,Iai->ai'         , cc_obj.h1p_eff, amp[2])
    res +=    -1.000000 * einsum('ji,aj->ai'         , cc_obj.h1e.oo, amp[0])
    res +=     1.000000 * einsum('ab,bi->ai'         , cc_obj.h1e.vv, amp[0])
    res +=     1.000000 * einsum('Iai,I->ai'         , cc_obj.h1p1e.vo, amp[1])
    res +=    -1.000000 * einsum('Iji,Iaj->ai'       , cc_obj.h1p1e.oo, amp[2])
    res +=     1.000000 * einsum('Iab,Ibi->ai'       , cc_obj.h1p1e.vv, amp[2])
    res +=    -1.000000 * einsum('jb,bi,aj->ai'      , cc_obj.h1e.ov, amp[0], amp[0])
    res +=    -1.000000 * einsum('Iji,aj,I->ai'      , cc_obj.h1p1e.oo, amp[0], amp[1])
    res +=     1.000000 * einsum('Iab,bi,I->ai'      , cc_obj.h1p1e.vv, amp[0], amp[1])
    res +=    -1.000000 * einsum('Ijb,aj,Ibi->ai'    , cc_obj.h1p1e.ov, amp[0], amp[2])
    res +=    -1.000000 * einsum('Ijb,bi,Iaj->ai'    , cc_obj.h1p1e.ov, amp[0], amp[2])
    res +=     1.000000 * einsum('Ijb,bj,Iai->ai'    , cc_obj.h1p1e.ov, amp[0], amp[2])
    res +=    -1.000000 * einsum('Ijb,bi,aj,I->ai'   , cc_obj.h1p1e.ov, amp[0], amp[0], amp[1])
    return res

def res_1(cc_obj, amp):
    """
    Generated by gen-cceqs.py at 2023-09-20 13:45:06
    Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    Hostname:     pauling001
    Git Branch:   pauling
    Git Commit:   f22052967d28d07b3f2de001f71f7a160db878c0

    amp[0]: a+_v a_o
    amp[1]: b+
    amp[2]: b+ a+_v a_o
    res   :  b

    ih = 0, ibra = 1, len(tmp.terms) = 1
    ih = 1, ibra = 1, len(tmp.terms) = 3
    ih = 2, ibra = 1, len(tmp.terms) = 1
    ih = 3, ibra = 1, len(tmp.terms) = 0
    """
    res  =     1.000000 * einsum('I->I'              , cc_obj.h1p_eff)
    res +=     1.000000 * einsum('IJ,J->I'           , cc_obj.h1p, amp[1])
    res +=     1.000000 * einsum('ia,Iai->I'         , cc_obj.h1e.ov, amp[2])
    res +=     1.000000 * einsum('Iia,ai->I'         , cc_obj.h1p1e.ov, amp[0])
    res +=     1.000000 * einsum('Jia,J,Iai->I'      , cc_obj.h1p1e.ov, amp[1], amp[2])
    return res

def res_2(cc_obj, amp):
    """
    Generated by gen-cceqs.py at 2023-09-20 13:45:06
    Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    Hostname:     pauling001
    Git Branch:   pauling
    Git Commit:   f22052967d28d07b3f2de001f71f7a160db878c0

    amp[0]: a+_v a_o
    amp[1]: b+
    amp[2]: b+ a+_v a_o
    res   :  b a+_o a_v

    ih = 0, ibra = 2, len(tmp.terms) = 1
    ih = 1, ibra = 2, len(tmp.terms) = 5
    ih = 2, ibra = 2, len(tmp.terms) = 8
    ih = 3, ibra = 2, len(tmp.terms) = 2
    ih = 4, ibra = 2, len(tmp.terms) = 0
    """
    res  =     1.000000 * einsum('Iai->Iai'          , cc_obj.h1p1e.vo)
    res +=    -1.000000 * einsum('ji,Iaj->Iai'       , cc_obj.h1e.oo, amp[2])
    res +=     1.000000 * einsum('ab,Ibi->Iai'       , cc_obj.h1e.vv, amp[2])
    res +=     1.000000 * einsum('IJ,Jai->Iai'       , cc_obj.h1p, amp[2])
    res +=    -1.000000 * einsum('Iji,aj->Iai'       , cc_obj.h1p1e.oo, amp[0])
    res +=     1.000000 * einsum('Iab,bi->Iai'       , cc_obj.h1p1e.vv, amp[0])
    res +=    -1.000000 * einsum('jb,aj,Ibi->Iai'    , cc_obj.h1e.ov, amp[0], amp[2])
    res +=    -1.000000 * einsum('jb,bi,Iaj->Iai'    , cc_obj.h1e.ov, amp[0], amp[2])
    res +=    -1.000000 * einsum('Ijb,bi,aj->Iai'    , cc_obj.h1p1e.ov, amp[0], amp[0])
    res +=    -1.000000 * einsum('Jji,J,Iaj->Iai'    , cc_obj.h1p1e.oo, amp[1], amp[2])
    res +=     1.000000 * einsum('Jab,J,Ibi->Iai'    , cc_obj.h1p1e.vv, amp[1], amp[2])
    res +=    -1.000000 * einsum('Jjb,Iaj,Jbi->Iai'  , cc_obj.h1p1e.ov, amp[2], amp[2])
    res +=    -1.000000 * einsum('Jjb,Ibi,Jaj->Iai'  , cc_obj.h1p1e.ov, amp[2], amp[2])
    res +=     1.000000 * einsum('Jjb,Ibj,Jai->Iai'  , cc_obj.h1p1e.ov, amp[2], amp[2])
    res +=    -1.000000 * einsum('Jjb,aj,J,Ibi->Iai' , cc_obj.h1p1e.ov, amp[0], amp[1], amp[2])
    res +=    -1.000000 * einsum('Jjb,bi,J,Iaj->Iai' , cc_obj.h1p1e.ov, amp[0], amp[1], amp[2])
    return res

def ene(cc_obj, amp):
    """
    Generated by gen-cceqs.py at 2023-09-20 13:45:13
    Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    Hostname:     pauling001
    Git Branch:   pauling
    Git Commit:   f22052967d28d07b3f2de001f71f7a160db878c0

    amp[0]: a+_v a_o
    amp[1]: b+
    amp[2]: b+ a+_v a_o

    ih = 0, ibra = 3, len(tmp.terms) = 0
    ih = 1, ibra = 3, len(tmp.terms) = 3
    ih = 2, ibra = 3, len(tmp.terms) = 1
    ih = 3, ibra = 3, len(tmp.terms) = 0
    """
    res  =     1.000000 * einsum('I,I->'             , cc_obj.h1p_eff, amp[1])
    res +=     1.000000 * einsum('ia,ai->'           , cc_obj.h1e.ov, amp[0])
    res +=     1.000000 * einsum('Iia,Iai->'         , cc_obj.h1p1e.ov, amp[2])
    res +=     1.000000 * einsum('Iia,ai,I->'        , cc_obj.h1p1e.ov, amp[0], amp[1])
    return res
