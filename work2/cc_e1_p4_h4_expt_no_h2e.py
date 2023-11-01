import numpy, functools
einsum = functools.partial(numpy.einsum, optimize=True)

def exp_nb_4(cc_obj=None, amp=None, lam=None):
    """
    Generated by gen-cceqs.py at 2023-10-31 16:27:23
    Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    Hostname:     pauling001
    Git Branch:   pauling
    Git Commit:   a256f996f2fbfa91ac721f723c393e84c7325eea

    amp[0]: a+_v a_o
    amp[1]: b+
    amp[2]: b+ a+_v a_o
    amp[3]: b+ b+
    amp[4]: b+ b+ a+_v a_o
    amp[5]: b+ b+ b+
    amp[6]: b+ b+ b+ a+_v a_o
    amp[7]: b+ b+ b+ b+
    amp[8]: b+ b+ b+ b+ a+_v a_o
    lam[0]: a+_o a_v
    lam[1]: b
    lam[2]: b a+_o a_v
    lam[3]: b b
    lam[4]: b b a+_o a_v
    lam[5]: b b b
    lam[6]: b b b a+_o a_v
    lam[7]: b b b b
    lam[8]: b b b b a+_o a_v

    """
    res  =     1.000000 * einsum('I,J->IJ'           , lam[1], amp[1])
    res +=     1.000000 * einsum('IJ,JK->IK'         , lam[3], amp[3])
    res +=     1.000000 * einsum('Iia,Jai->IJ'       , lam[2], amp[2])
    res +=     0.500000 * einsum('IJK,JKL->IL'       , lam[5], amp[5])
    res +=     1.000000 * einsum('IJia,JKai->IK'     , lam[4], amp[4])
    res +=     0.166667 * einsum('IJKL,JKLM->IM'     , lam[7], amp[7])
    res +=     0.500000 * einsum('IJKia,JKLai->IL'   , lam[6], amp[6])
    res +=     0.166667 * einsum('IJKLia,JKLMai->IM' , lam[8], amp[8])
    return res

def exp_xb_4(cc_obj=None, amp=None, lam=None):
    """
    Generated by gen-cceqs.py at 2023-10-31 17:12:43
    Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    Hostname:     pauling001
    Git Branch:   pauling
    Git Commit:   a256f996f2fbfa91ac721f723c393e84c7325eea

    amp[0]: a+_v a_o
    amp[1]: b+
    amp[2]: b+ a+_v a_o
    amp[3]: b+ b+
    amp[4]: b+ b+ a+_v a_o
    amp[5]: b+ b+ b+
    amp[6]: b+ b+ b+ a+_v a_o
    amp[7]: b+ b+ b+ b+
    amp[8]: b+ b+ b+ b+ a+_v a_o
    lam[0]: a+_o a_v
    lam[1]: b
    lam[2]: b a+_o a_v
    lam[3]: b b
    lam[4]: b b a+_o a_v
    lam[5]: b b b
    lam[6]: b b b a+_o a_v
    lam[7]: b b b b
    lam[8]: b b b b a+_o a_v

    """
    res  =     1.000000 * einsum('I->I'              , amp[1])
    res +=     1.000000 * einsum('I->I'              , lam[1])
    res +=     1.000000 * einsum('I,IJ->J'           , lam[1], amp[3])
    res +=     1.000000 * einsum('ia,Iai->I'         , lam[0], amp[2])
    res +=     0.500000 * einsum('IJ,IJK->K'         , lam[3], amp[5])
    res +=     1.000000 * einsum('Iia,IJai->J'       , lam[2], amp[4])
    res +=     0.166667 * einsum('IJK,IJKL->L'       , lam[5], amp[7])
    res +=     0.500000 * einsum('IJia,IJKai->K'     , lam[4], amp[6])
    res +=     0.166667 * einsum('IJKia,IJKLai->L'   , lam[6], amp[8])
    return res

def exp_na_ov_4(cc_obj=None, amp=None, lam=None):
    """
    Generated by gen-cceqs.py at 2023-10-31 17:16:49
    Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    Hostname:     pauling001
    Git Branch:   pauling
    Git Commit:   a256f996f2fbfa91ac721f723c393e84c7325eea

    amp[0]: a+_v a_o
    amp[1]: b+
    amp[2]: b+ a+_v a_o
    amp[3]: b+ b+
    amp[4]: b+ b+ a+_v a_o
    amp[5]: b+ b+ b+
    amp[6]: b+ b+ b+ a+_v a_o
    amp[7]: b+ b+ b+ b+
    amp[8]: b+ b+ b+ b+ a+_v a_o
    lam[0]: a+_o a_v
    lam[1]: b
    lam[2]: b a+_o a_v
    lam[3]: b b
    lam[4]: b b a+_o a_v
    lam[5]: b b b
    lam[6]: b b b a+_o a_v
    lam[7]: b b b b
    lam[8]: b b b b a+_o a_v

    """
    res  =     1.000000 * einsum('ia->ia'            , lam[0])
    return res

def exp_na_ov_xb_4(cc_obj=None, amp=None, lam=None):
    """
    Generated by gen-cceqs.py at 2023-10-31 17:30:02
    Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    Hostname:     pauling001
    Git Branch:   pauling
    Git Commit:   a256f996f2fbfa91ac721f723c393e84c7325eea

    amp[0]: a+_v a_o
    amp[1]: b+
    amp[2]: b+ a+_v a_o
    amp[3]: b+ b+
    amp[4]: b+ b+ a+_v a_o
    amp[5]: b+ b+ b+
    amp[6]: b+ b+ b+ a+_v a_o
    amp[7]: b+ b+ b+ b+
    amp[8]: b+ b+ b+ b+ a+_v a_o
    lam[0]: a+_o a_v
    lam[1]: b
    lam[2]: b a+_o a_v
    lam[3]: b b
    lam[4]: b b a+_o a_v
    lam[5]: b b b
    lam[6]: b b b a+_o a_v
    lam[7]: b b b b
    lam[8]: b b b b a+_o a_v

    """
    res  =     1.000000 * einsum('Iia->iaI'          , lam[2])
    res +=     1.000000 * einsum('ia,I->iaI'         , lam[0], amp[1])
    res +=     1.000000 * einsum('Iia,IJ->iaJ'       , lam[2], amp[3])
    res +=     0.500000 * einsum('IJia,IJK->iaK'     , lam[4], amp[5])
    res +=     0.166667 * einsum('IJKia,IJKL->iaL'   , lam[6], amp[7])
    return res

def exp_na_vo_4(cc_obj=None, amp=None, lam=None):
    """
    Generated by gen-cceqs.py at 2023-10-31 17:55:39
    Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    Hostname:     pauling001
    Git Branch:   pauling
    Git Commit:   a256f996f2fbfa91ac721f723c393e84c7325eea

    amp[0]: a+_v a_o
    amp[1]: b+
    amp[2]: b+ a+_v a_o
    amp[3]: b+ b+
    amp[4]: b+ b+ a+_v a_o
    amp[5]: b+ b+ b+
    amp[6]: b+ b+ b+ a+_v a_o
    amp[7]: b+ b+ b+ b+
    amp[8]: b+ b+ b+ b+ a+_v a_o
    lam[0]: a+_o a_v
    lam[1]: b
    lam[2]: b a+_o a_v
    lam[3]: b b
    lam[4]: b b a+_o a_v
    lam[5]: b b b
    lam[6]: b b b a+_o a_v
    lam[7]: b b b b
    lam[8]: b b b b a+_o a_v

    """
    res  =     1.000000 * einsum('ai->ai'            , amp[0])
    res +=     1.000000 * einsum('I,Iai->ai'         , lam[1], amp[2])
    res +=     0.500000 * einsum('IJ,IJai->ai'       , lam[3], amp[4])
    res +=     0.166667 * einsum('IJK,IJKai->ai'     , lam[5], amp[6])
    res +=     0.041667 * einsum('IJKL,IJKLai->ai'   , lam[7], amp[8])
    res +=    -1.000000 * einsum('ia,bi,aj->bj'      , lam[0], amp[0], amp[0])
    res +=    -1.000000 * einsum('Iia,aj,Ibi->bj'    , lam[2], amp[0], amp[2])
    res +=    -1.000000 * einsum('Iia,bi,Iaj->bj'    , lam[2], amp[0], amp[2])
    res +=    -0.500000 * einsum('IJia,aj,IJbi->bj'  , lam[4], amp[0], amp[4])
    res +=    -0.500000 * einsum('IJia,bi,IJaj->bj'  , lam[4], amp[0], amp[4])
    res +=    -1.000000 * einsum('IJia,Ibi,Jaj->bj'  , lam[4], amp[2], amp[2])
    res +=    -0.166667 * einsum('IJKia,aj,IJKbi->bj', lam[6], amp[0], amp[6])
    res +=    -0.166667 * einsum('IJKia,bi,IJKaj->bj', lam[6], amp[0], amp[6])
    res +=    -0.500000 * einsum('IJKia,Iaj,JKbi->bj', lam[6], amp[2], amp[4])
    res +=    -0.500000 * einsum('IJKia,Ibi,JKaj->bj', lam[6], amp[2], amp[4])
    res +=    -0.041667 * einsum('IJKLia,aj,IJKLbi->bj', lam[8], amp[0], amp[8])
    res +=    -0.041667 * einsum('IJKLia,bi,IJKLaj->bj', lam[8], amp[0], amp[8])
    res +=    -0.166667 * einsum('IJKLia,Iaj,JKLbi->bj', lam[8], amp[2], amp[6])
    res +=    -0.166667 * einsum('IJKLia,Ibi,JKLaj->bj', lam[8], amp[2], amp[6])
    res +=    -0.250000 * einsum('IJKLia,IJbi,KLaj->bj', lam[8], amp[4], amp[4])
    return res

def exp_na_vo_xb_4(cc_obj=None, amp=None, lam=None):
    """
    Generated by gen-cceqs.py at 2023-11-01 08:03:41
    Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    Hostname:     pauling001
    Git Branch:   pauling
    Git Commit:   c7dbb7ec133d24a494d4610310481784ff325eae

    amp[0]: a+_v a_o
    amp[1]: b+
    amp[2]: b+ a+_v a_o
    amp[3]: b+ b+
    amp[4]: b+ b+ a+_v a_o
    amp[5]: b+ b+ b+
    amp[6]: b+ b+ b+ a+_v a_o
    amp[7]: b+ b+ b+ b+
    amp[8]: b+ b+ b+ b+ a+_v a_o
    lam[0]: a+_o a_v
    lam[1]: b
    lam[2]: b a+_o a_v
    lam[3]: b b
    lam[4]: b b a+_o a_v
    lam[5]: b b b
    lam[6]: b b b a+_o a_v
    lam[7]: b b b b
    lam[8]: b b b b a+_o a_v

    """
    res  =     1.000000 * einsum('Iai->aiI'          , amp[2])
    res +=     1.000000 * einsum('I,ai->aiI'         , lam[1], amp[0])
    res +=     1.000000 * einsum('ai,I->aiI'         , amp[0], amp[1])
    res +=     1.000000 * einsum('I,IJai->aiJ'       , lam[1], amp[4])
    res +=     1.000000 * einsum('IJ,Jai->aiI'       , lam[3], amp[2])
    res +=     0.500000 * einsum('IJ,IJKai->aiK'     , lam[3], amp[6])
    res +=     0.500000 * einsum('IJK,JKai->aiI'     , lam[5], amp[4])
    res +=     0.166667 * einsum('IJK,IJKLai->aiL'   , lam[5], amp[8])
    res +=     0.166667 * einsum('IJKL,JKLai->aiI'   , lam[7], amp[6])
    res +=     1.000000 * einsum('I,J,Iai->aiJ'      , lam[1], amp[1], amp[2])
    res +=     1.000000 * einsum('I,ai,IJ->aiJ'      , lam[1], amp[0], amp[3])
    res +=    -1.000000 * einsum('ia,aj,Ibi->bjI'    , lam[0], amp[0], amp[2])
    res +=    -1.000000 * einsum('ia,bi,Iaj->bjI'    , lam[0], amp[0], amp[2])
    res +=     1.000000 * einsum('ia,bj,Iai->bjI'    , lam[0], amp[0], amp[2])
    res +=     0.500000 * einsum('IJ,K,IJai->aiK'    , lam[3], amp[1], amp[4])
    res +=     0.500000 * einsum('IJ,ai,IJK->aiK'    , lam[3], amp[0], amp[5])
    res +=     1.000000 * einsum('IJ,Iai,JK->aiK'    , lam[3], amp[2], amp[3])
    res +=    -1.000000 * einsum('Iia,bi,aj->bjI'    , lam[2], amp[0], amp[0])
    res +=    -1.000000 * einsum('Iia,aj,IJbi->bjJ'  , lam[2], amp[0], amp[4])
    res +=    -1.000000 * einsum('Iia,bi,IJaj->bjJ'  , lam[2], amp[0], amp[4])
    res +=     1.000000 * einsum('Iia,bj,IJai->bjJ'  , lam[2], amp[0], amp[4])
    res +=    -1.000000 * einsum('Iia,Iaj,Jbi->bjJ'  , lam[2], amp[2], amp[2])
    res +=    -1.000000 * einsum('Iia,Ibi,Jaj->bjJ'  , lam[2], amp[2], amp[2])
    res +=     1.000000 * einsum('Iia,Ibj,Jai->bjJ'  , lam[2], amp[2], amp[2])
    res +=     0.166667 * einsum('IJK,L,IJKai->aiL'  , lam[5], amp[1], amp[6])
    res +=     0.166667 * einsum('IJK,ai,IJKL->aiL'  , lam[5], amp[0], amp[7])
    res +=     0.500000 * einsum('IJK,IL,JKai->aiL'  , lam[5], amp[3], amp[4])
    res +=     0.500000 * einsum('IJK,Iai,JKL->aiL'  , lam[5], amp[2], amp[5])
    res +=    -1.000000 * einsum('IJia,aj,Jbi->bjI'  , lam[4], amp[0], amp[2])
    res +=    -1.000000 * einsum('IJia,bi,Jaj->bjI'  , lam[4], amp[0], amp[2])
    res +=    -0.500000 * einsum('IJia,aj,IJKbi->bjK', lam[4], amp[0], amp[6])
    res +=    -0.500000 * einsum('IJia,bi,IJKaj->bjK', lam[4], amp[0], amp[6])
    res +=     0.500000 * einsum('IJia,bj,IJKai->bjK', lam[4], amp[0], amp[6])
    res +=    -1.000000 * einsum('IJia,Iaj,JKbi->bjK', lam[4], amp[2], amp[4])
    res +=    -1.000000 * einsum('IJia,Ibi,JKaj->bjK', lam[4], amp[2], amp[4])
    res +=     1.000000 * einsum('IJia,Ibj,JKai->bjK', lam[4], amp[2], amp[4])
    res +=     0.500000 * einsum('IJia,Kai,IJbj->bjK', lam[4], amp[2], amp[4])
    res +=    -0.500000 * einsum('IJia,Kaj,IJbi->bjK', lam[4], amp[2], amp[4])
    res +=    -0.500000 * einsum('IJia,Kbi,IJaj->bjK', lam[4], amp[2], amp[4])
    res +=     0.041667 * einsum('IJKL,M,IJKLai->aiM', lam[7], amp[1], amp[8])
    res +=     0.166667 * einsum('IJKL,IM,JKLai->aiM', lam[7], amp[3], amp[6])
    res +=     0.166667 * einsum('IJKL,Iai,JKLM->aiM', lam[7], amp[2], amp[7])
    res +=     0.250000 * einsum('IJKL,IJai,KLM->aiM', lam[7], amp[4], amp[5])
    res +=    -0.500000 * einsum('IJKia,aj,JKbi->bjI', lam[6], amp[0], amp[4])
    res +=    -0.500000 * einsum('IJKia,bi,JKaj->bjI', lam[6], amp[0], amp[4])
    res +=    -1.000000 * einsum('IJKia,Jbi,Kaj->bjI', lam[6], amp[2], amp[2])
    res +=    -0.166667 * einsum('IJKia,aj,IJKLbi->bjL', lam[6], amp[0], amp[8])
    res +=    -0.166667 * einsum('IJKia,bi,IJKLaj->bjL', lam[6], amp[0], amp[8])
    res +=     0.166667 * einsum('IJKia,bj,IJKLai->bjL', lam[6], amp[0], amp[8])
    res +=    -0.500000 * einsum('IJKia,Iaj,JKLbi->bjL', lam[6], amp[2], amp[6])
    res +=    -0.500000 * einsum('IJKia,Ibi,JKLaj->bjL', lam[6], amp[2], amp[6])
    res +=     0.500000 * einsum('IJKia,Ibj,JKLai->bjL', lam[6], amp[2], amp[6])
    res +=     0.166667 * einsum('IJKia,Lai,IJKbj->bjL', lam[6], amp[2], amp[6])
    res +=    -0.166667 * einsum('IJKia,Laj,IJKbi->bjL', lam[6], amp[2], amp[6])
    res +=    -0.166667 * einsum('IJKia,Lbi,IJKaj->bjL', lam[6], amp[2], amp[6])
    res +=    -0.500000 * einsum('IJKia,IJaj,KLbi->bjL', lam[6], amp[4], amp[4])
    res +=    -0.500000 * einsum('IJKia,IJbi,KLaj->bjL', lam[6], amp[4], amp[4])
    res +=     0.500000 * einsum('IJKia,IJbj,KLai->bjL', lam[6], amp[4], amp[4])
    res +=    -0.166667 * einsum('IJKLia,aj,JKLbi->bjI', lam[8], amp[0], amp[6])
    res +=    -0.166667 * einsum('IJKLia,bi,JKLaj->bjI', lam[8], amp[0], amp[6])
    res +=    -0.500000 * einsum('IJKLia,Jaj,KLbi->bjI', lam[8], amp[2], amp[4])
    res +=    -0.500000 * einsum('IJKLia,Jbi,KLaj->bjI', lam[8], amp[2], amp[4])
    res +=    -0.166667 * einsum('IJKLia,Iaj,JKLMbi->bjM', lam[8], amp[2], amp[8])
    res +=    -0.166667 * einsum('IJKLia,Ibi,JKLMaj->bjM', lam[8], amp[2], amp[8])
    res +=     0.166667 * einsum('IJKLia,Ibj,JKLMai->bjM', lam[8], amp[2], amp[8])
    res +=     0.041667 * einsum('IJKLia,Mai,IJKLbj->bjM', lam[8], amp[2], amp[8])
    res +=    -0.041667 * einsum('IJKLia,Maj,IJKLbi->bjM', lam[8], amp[2], amp[8])
    res +=    -0.041667 * einsum('IJKLia,Mbi,IJKLaj->bjM', lam[8], amp[2], amp[8])
    res +=    -0.250000 * einsum('IJKLia,IJaj,KLMbi->bjM', lam[8], amp[4], amp[6])
    res +=    -0.250000 * einsum('IJKLia,IJbi,KLMaj->bjM', lam[8], amp[4], amp[6])
    res +=     0.250000 * einsum('IJKLia,IJbj,KLMai->bjM', lam[8], amp[4], amp[6])
    res +=     0.166667 * einsum('IJKLia,IMai,JKLbj->bjM', lam[8], amp[4], amp[6])
    res +=    -0.166667 * einsum('IJKLia,IMaj,JKLbi->bjM', lam[8], amp[4], amp[6])
    res +=    -0.166667 * einsum('IJKLia,IMbi,JKLaj->bjM', lam[8], amp[4], amp[6])
    res +=    -1.000000 * einsum('ia,bi,aj,I->bjI'   , lam[0], amp[0], amp[0], amp[1])
    res +=    -1.000000 * einsum('Iia,aj,J,Ibi->bjJ' , lam[2], amp[0], amp[1], amp[2])
    res +=    -1.000000 * einsum('Iia,bi,J,Iaj->bjJ' , lam[2], amp[0], amp[1], amp[2])
    res +=    -1.000000 * einsum('Iia,bi,aj,IJ->bjJ' , lam[2], amp[0], amp[0], amp[3])
    res +=    -1.000000 * einsum('IJia,K,Ibi,Jaj->bjK', lam[4], amp[1], amp[2], amp[2])
    res +=    -0.500000 * einsum('IJia,aj,K,IJbi->bjK', lam[4], amp[0], amp[1], amp[4])
    res +=    -1.000000 * einsum('IJia,aj,Ibi,JK->bjK', lam[4], amp[0], amp[2], amp[3])
    res +=    -0.500000 * einsum('IJia,bi,K,IJaj->bjK', lam[4], amp[0], amp[1], amp[4])
    res +=    -0.500000 * einsum('IJia,bi,aj,IJK->bjK', lam[4], amp[0], amp[0], amp[5])
    res +=    -1.000000 * einsum('IJia,bi,Iaj,JK->bjK', lam[4], amp[0], amp[2], amp[3])
    res +=    -0.500000 * einsum('IJKia,L,Iaj,JKbi->bjL', lam[6], amp[1], amp[2], amp[4])
    res +=    -0.500000 * einsum('IJKia,L,Ibi,JKaj->bjL', lam[6], amp[1], amp[2], amp[4])
    res +=    -0.166667 * einsum('IJKia,aj,L,IJKbi->bjL', lam[6], amp[0], amp[1], amp[6])
    res +=    -0.500000 * einsum('IJKia,aj,IL,JKbi->bjL', lam[6], amp[0], amp[3], amp[4])
    res +=    -0.500000 * einsum('IJKia,aj,Ibi,JKL->bjL', lam[6], amp[0], amp[2], amp[5])
    res +=    -0.166667 * einsum('IJKia,bi,L,IJKaj->bjL', lam[6], amp[0], amp[1], amp[6])
    res +=    -0.166667 * einsum('IJKia,bi,aj,IJKL->bjL', lam[6], amp[0], amp[0], amp[7])
    res +=    -0.500000 * einsum('IJKia,bi,IL,JKaj->bjL', lam[6], amp[0], amp[3], amp[4])
    res +=    -0.500000 * einsum('IJKia,bi,Iaj,JKL->bjL', lam[6], amp[0], amp[2], amp[5])
    res +=    -1.000000 * einsum('IJKia,Ibi,Jaj,KL->bjL', lam[6], amp[2], amp[2], amp[3])
    res +=    -0.166667 * einsum('IJKLia,M,Iaj,JKLbi->bjM', lam[8], amp[1], amp[2], amp[6])
    res +=    -0.166667 * einsum('IJKLia,M,Ibi,JKLaj->bjM', lam[8], amp[1], amp[2], amp[6])
    res +=    -0.250000 * einsum('IJKLia,M,IJbi,KLaj->bjM', lam[8], amp[1], amp[4], amp[4])
    res +=    -0.041667 * einsum('IJKLia,aj,M,IJKLbi->bjM', lam[8], amp[0], amp[1], amp[8])
    res +=    -0.166667 * einsum('IJKLia,aj,IM,JKLbi->bjM', lam[8], amp[0], amp[3], amp[6])
    res +=    -0.166667 * einsum('IJKLia,aj,Ibi,JKLM->bjM', lam[8], amp[0], amp[2], amp[7])
    res +=    -0.250000 * einsum('IJKLia,aj,IJbi,KLM->bjM', lam[8], amp[0], amp[4], amp[5])
    res +=    -0.041667 * einsum('IJKLia,bi,M,IJKLaj->bjM', lam[8], amp[0], amp[1], amp[8])
    res +=    -0.166667 * einsum('IJKLia,bi,IM,JKLaj->bjM', lam[8], amp[0], amp[3], amp[6])
    res +=    -0.166667 * einsum('IJKLia,bi,Iaj,JKLM->bjM', lam[8], amp[0], amp[2], amp[7])
    res +=    -0.250000 * einsum('IJKLia,bi,IJaj,KLM->bjM', lam[8], amp[0], amp[4], amp[5])
    res +=    -0.500000 * einsum('IJKLia,Iaj,JM,KLbi->bjM', lam[8], amp[2], amp[3], amp[4])
    res +=    -0.500000 * einsum('IJKLia,Ibi,JM,KLaj->bjM', lam[8], amp[2], amp[3], amp[4])
    res +=    -0.500000 * einsum('IJKLia,Ibi,Jaj,KLM->bjM', lam[8], amp[2], amp[2], amp[5])
    return res

def exp_na_oo_4(cc_obj=None, amp=None, lam=None):
    """
    Generated by gen-cceqs.py at 2023-11-01 08:13:06
    Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    Hostname:     pauling001
    Git Branch:   pauling
    Git Commit:   c7dbb7ec133d24a494d4610310481784ff325eae

    amp[0]: a+_v a_o
    amp[1]: b+
    amp[2]: b+ a+_v a_o
    amp[3]: b+ b+
    amp[4]: b+ b+ a+_v a_o
    amp[5]: b+ b+ b+
    amp[6]: b+ b+ b+ a+_v a_o
    amp[7]: b+ b+ b+ b+
    amp[8]: b+ b+ b+ b+ a+_v a_o
    lam[0]: a+_o a_v
    lam[1]: b
    lam[2]: b a+_o a_v
    lam[3]: b b
    lam[4]: b b a+_o a_v
    lam[5]: b b b
    lam[6]: b b b a+_o a_v
    lam[7]: b b b b
    lam[8]: b b b b a+_o a_v

    """
    res  =     1.000000 * einsum('ji->ij'            , cc_obj.delta.oo)
    res +=    -1.000000 * einsum('ia,aj->ij'         , lam[0], amp[0])
    res +=    -1.000000 * einsum('Iia,Iaj->ij'       , lam[2], amp[2])
    res +=    -0.500000 * einsum('IJia,IJaj->ij'     , lam[4], amp[4])
    res +=    -0.166667 * einsum('IJKia,IJKaj->ij'   , lam[6], amp[6])
    res +=    -0.041667 * einsum('IJKLia,IJKLaj->ij' , lam[8], amp[8])
    return res

def exp_na_oo_xb_4(cc_obj=None, amp=None, lam=None):
    """
    Generated by gen-cceqs.py at 2023-11-01 11:41:55
    Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    Hostname:     pauling001
    Git Branch:   pauling
    Git Commit:   c7dbb7ec133d24a494d4610310481784ff325eae

    amp[0]: a+_v a_o
    amp[1]: b+
    amp[2]: b+ a+_v a_o
    amp[3]: b+ b+
    amp[4]: b+ b+ a+_v a_o
    amp[5]: b+ b+ b+
    amp[6]: b+ b+ b+ a+_v a_o
    amp[7]: b+ b+ b+ b+
    amp[8]: b+ b+ b+ b+ a+_v a_o
    lam[0]: a+_o a_v
    lam[1]: b
    lam[2]: b a+_o a_v
    lam[3]: b b
    lam[4]: b b a+_o a_v
    lam[5]: b b b
    lam[6]: b b b a+_o a_v
    lam[7]: b b b b
    lam[8]: b b b b a+_o a_v

    """
    res  =     1.000000 * einsum('I,ji->ijI'         , lam[1], cc_obj.delta.oo)
    res +=     1.000000 * einsum('I,ji->ijI'         , amp[1], cc_obj.delta.oo)
    res +=    -1.000000 * einsum('ia,Iaj->ijI'       , lam[0], amp[2])
    res +=    -1.000000 * einsum('Iia,aj->ijI'       , lam[2], amp[0])
    res +=    -1.000000 * einsum('Iia,IJaj->ijJ'     , lam[2], amp[4])
    res +=    -1.000000 * einsum('IJia,Jaj->ijI'     , lam[4], amp[2])
    res +=    -0.500000 * einsum('IJia,IJKaj->ijK'   , lam[4], amp[6])
    res +=    -0.500000 * einsum('IJKia,JKaj->ijI'   , lam[6], amp[4])
    res +=    -0.166667 * einsum('IJKia,IJKLaj->ijL' , lam[6], amp[8])
    res +=    -0.166667 * einsum('IJKLia,JKLaj->ijI' , lam[8], amp[6])
    res +=     1.000000 * einsum('I,IJ,ji->ijJ'      , lam[1], amp[3], cc_obj.delta.oo)
    res +=    -1.000000 * einsum('ia,aj,I->ijI'      , lam[0], amp[0], amp[1])
    res +=     1.000000 * einsum('ia,Iai,kj->jkI'    , lam[0], amp[2], cc_obj.delta.oo)
    res +=     0.500000 * einsum('IJ,IJK,ji->ijK'    , lam[3], amp[5], cc_obj.delta.oo)
    res +=    -1.000000 * einsum('Iia,J,Iaj->ijJ'    , lam[2], amp[1], amp[2])
    res +=    -1.000000 * einsum('Iia,aj,IJ->ijJ'    , lam[2], amp[0], amp[3])
    res +=     1.000000 * einsum('Iia,IJai,kj->jkJ'  , lam[2], amp[4], cc_obj.delta.oo)
    res +=     0.166667 * einsum('IJK,IJKL,ji->ijL'  , lam[5], amp[7], cc_obj.delta.oo)
    res +=    -0.500000 * einsum('IJia,K,IJaj->ijK'  , lam[4], amp[1], amp[4])
    res +=    -0.500000 * einsum('IJia,aj,IJK->ijK'  , lam[4], amp[0], amp[5])
    res +=    -1.000000 * einsum('IJia,Iaj,JK->ijK'  , lam[4], amp[2], amp[3])
    res +=     0.500000 * einsum('IJia,IJKai,kj->jkK', lam[4], amp[6], cc_obj.delta.oo)
    res +=    -0.166667 * einsum('IJKia,L,IJKaj->ijL', lam[6], amp[1], amp[6])
    res +=    -0.166667 * einsum('IJKia,aj,IJKL->ijL', lam[6], amp[0], amp[7])
    res +=    -0.500000 * einsum('IJKia,IL,JKaj->ijL', lam[6], amp[3], amp[4])
    res +=    -0.500000 * einsum('IJKia,Iaj,JKL->ijL', lam[6], amp[2], amp[5])
    res +=     0.166667 * einsum('IJKia,IJKLai,kj->jkL', lam[6], amp[8], cc_obj.delta.oo)
    res +=    -0.041667 * einsum('IJKLia,M,IJKLaj->ijM', lam[8], amp[1], amp[8])
    res +=    -0.166667 * einsum('IJKLia,IM,JKLaj->ijM', lam[8], amp[3], amp[6])
    res +=    -0.166667 * einsum('IJKLia,Iaj,JKLM->ijM', lam[8], amp[2], amp[7])
    res +=    -0.250000 * einsum('IJKLia,IJaj,KLM->ijM', lam[8], amp[4], amp[5])
    return res

def exp_na_vv_4(cc_obj=None, amp=None, lam=None):
    """
    Generated by gen-cceqs.py at 2023-11-01 11:45:37
    Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    Hostname:     pauling001
    Git Branch:   pauling
    Git Commit:   c7dbb7ec133d24a494d4610310481784ff325eae

    amp[0]: a+_v a_o
    amp[1]: b+
    amp[2]: b+ a+_v a_o
    amp[3]: b+ b+
    amp[4]: b+ b+ a+_v a_o
    amp[5]: b+ b+ b+
    amp[6]: b+ b+ b+ a+_v a_o
    amp[7]: b+ b+ b+ b+
    amp[8]: b+ b+ b+ b+ a+_v a_o
    lam[0]: a+_o a_v
    lam[1]: b
    lam[2]: b a+_o a_v
    lam[3]: b b
    lam[4]: b b a+_o a_v
    lam[5]: b b b
    lam[6]: b b b a+_o a_v
    lam[7]: b b b b
    lam[8]: b b b b a+_o a_v

    """
    res  =     1.000000 * einsum('ia,bi->ba'         , lam[0], amp[0])
    res +=     1.000000 * einsum('Iia,Ibi->ba'       , lam[2], amp[2])
    res +=     0.500000 * einsum('IJia,IJbi->ba'     , lam[4], amp[4])
    res +=     0.166667 * einsum('IJKia,IJKbi->ba'   , lam[6], amp[6])
    res +=     0.041667 * einsum('IJKLia,IJKLbi->ba' , lam[8], amp[8])
    return res

def exp_na_vv_xb_4(cc_obj=None, amp=None, lam=None):
    """
    Generated by gen-cceqs.py at 2023-11-01 12:16:53
    Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    Hostname:     pauling001
    Git Branch:   pauling
    Git Commit:   c7dbb7ec133d24a494d4610310481784ff325eae

    amp[0]: a+_v a_o
    amp[1]: b+
    amp[2]: b+ a+_v a_o
    amp[3]: b+ b+
    amp[4]: b+ b+ a+_v a_o
    amp[5]: b+ b+ b+
    amp[6]: b+ b+ b+ a+_v a_o
    amp[7]: b+ b+ b+ b+
    amp[8]: b+ b+ b+ b+ a+_v a_o
    lam[0]: a+_o a_v
    lam[1]: b
    lam[2]: b a+_o a_v
    lam[3]: b b
    lam[4]: b b a+_o a_v
    lam[5]: b b b
    lam[6]: b b b a+_o a_v
    lam[7]: b b b b
    lam[8]: b b b b a+_o a_v

    """
    res  =     1.000000 * einsum('ia,Ibi->baI'       , lam[0], amp[2])
    res +=     1.000000 * einsum('Iia,bi->baI'       , lam[2], amp[0])
    res +=     1.000000 * einsum('Iia,IJbi->baJ'     , lam[2], amp[4])
    res +=     1.000000 * einsum('IJia,Jbi->baI'     , lam[4], amp[2])
    res +=     0.500000 * einsum('IJia,IJKbi->baK'   , lam[4], amp[6])
    res +=     0.500000 * einsum('IJKia,JKbi->baI'   , lam[6], amp[4])
    res +=     0.166667 * einsum('IJKia,IJKLbi->baL' , lam[6], amp[8])
    res +=     0.166667 * einsum('IJKLia,JKLbi->baI' , lam[8], amp[6])
    res +=     1.000000 * einsum('ia,bi,I->baI'      , lam[0], amp[0], amp[1])
    res +=     1.000000 * einsum('Iia,J,Ibi->baJ'    , lam[2], amp[1], amp[2])
    res +=     1.000000 * einsum('Iia,bi,IJ->baJ'    , lam[2], amp[0], amp[3])
    res +=     0.500000 * einsum('IJia,K,IJbi->baK'  , lam[4], amp[1], amp[4])
    res +=     0.500000 * einsum('IJia,bi,IJK->baK'  , lam[4], amp[0], amp[5])
    res +=     1.000000 * einsum('IJia,Ibi,JK->baK'  , lam[4], amp[2], amp[3])
    res +=     0.166667 * einsum('IJKia,L,IJKbi->baL', lam[6], amp[1], amp[6])
    res +=     0.166667 * einsum('IJKia,bi,IJKL->baL', lam[6], amp[0], amp[7])
    res +=     0.500000 * einsum('IJKia,IL,JKbi->baL', lam[6], amp[3], amp[4])
    res +=     0.500000 * einsum('IJKia,Ibi,JKL->baL', lam[6], amp[2], amp[5])
    res +=     0.041667 * einsum('IJKLia,M,IJKLbi->baM', lam[8], amp[1], amp[8])
    res +=     0.166667 * einsum('IJKLia,IM,JKLbi->baM', lam[8], amp[3], amp[6])
    res +=     0.166667 * einsum('IJKLia,Ibi,JKLM->baM', lam[8], amp[2], amp[7])
    res +=     0.250000 * einsum('IJKLia,IJbi,KLM->baM', lam[8], amp[4], amp[5])
    return res
