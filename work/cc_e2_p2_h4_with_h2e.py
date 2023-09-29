import numpy, functools
einsum = functools.partial(numpy.einsum, optimize=True)

def res_0(cc_obj, amp):
    """
    Generated by gen-cceqs.py at 2023-09-20 12:31:45
    Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    Hostname:     pauling001
    Git Branch:   pauling
    Git Commit:   323903bb20c46a422c99b2415498f89681509a89

    amp[0]: a+_v a_o
    amp[1]: a+_v a+_v a_o a_o
    amp[2]: b+
    amp[3]: b+ a+_v a_o
    amp[4]: b+ b+
    amp[5]: b+ b+ a+_v a_o
    res   :  a+_o a_v

    ih = 0, ibra = 0, len(tmp.terms) = 1
    ih = 1, ibra = 0, len(tmp.terms) = 10
    ih = 2, ibra = 0, len(tmp.terms) = 12
    ih = 3, ibra = 0, len(tmp.terms) = 2
    ih = 4, ibra = 0, len(tmp.terms) = 0
    """
    res  =     1.000000 * einsum('ai->ai'            , cc_obj.h1e.vo)
    res +=     1.000000 * einsum('I,Iai->ai'         , cc_obj.h1p_eff, amp[3])
    res +=    -1.000000 * einsum('ji,aj->ai'         , cc_obj.h1e.oo, amp[0])
    res +=     1.000000 * einsum('ab,bi->ai'         , cc_obj.h1e.vv, amp[0])
    res +=     1.000000 * einsum('Iai,I->ai'         , cc_obj.h1p1e.vo, amp[2])
    res +=    -1.000000 * einsum('jb,abji->ai'       , cc_obj.h1e.ov, amp[1])
    res +=    -1.000000 * einsum('Iji,Iaj->ai'       , cc_obj.h1p1e.oo, amp[3])
    res +=     1.000000 * einsum('Iab,Ibi->ai'       , cc_obj.h1p1e.vv, amp[3])
    res +=    -1.000000 * einsum('jaib,bj->ai'       , cc_obj.h2e.ovov, amp[0])
    res +=     0.500000 * einsum('jkib,abkj->ai'     , cc_obj.h2e.ooov, amp[1])
    res +=    -0.500000 * einsum('jabc,cbji->ai'     , cc_obj.h2e.ovvv, amp[1])
    res +=    -1.000000 * einsum('jb,bi,aj->ai'      , cc_obj.h1e.ov, amp[0], amp[0])
    res +=    -1.000000 * einsum('Iji,aj,I->ai'      , cc_obj.h1p1e.oo, amp[0], amp[2])
    res +=     1.000000 * einsum('Iab,bi,I->ai'      , cc_obj.h1p1e.vv, amp[0], amp[2])
    res +=    -1.000000 * einsum('Ijb,aj,Ibi->ai'    , cc_obj.h1p1e.ov, amp[0], amp[3])
    res +=    -1.000000 * einsum('Ijb,bi,Iaj->ai'    , cc_obj.h1p1e.ov, amp[0], amp[3])
    res +=     1.000000 * einsum('Ijb,bj,Iai->ai'    , cc_obj.h1p1e.ov, amp[0], amp[3])
    res +=    -1.000000 * einsum('Ijb,abji,I->ai'    , cc_obj.h1p1e.ov, amp[1], amp[2])
    res +=    -1.000000 * einsum('jkib,aj,bk->ai'    , cc_obj.h2e.ooov, amp[0], amp[0])
    res +=     1.000000 * einsum('jabc,ci,bj->ai'    , cc_obj.h2e.ovvv, amp[0], amp[0])
    res +=    -0.500000 * einsum('jkbc,aj,cbki->ai'  , cc_obj.h2e.oovv, amp[0], amp[1])
    res +=    -0.500000 * einsum('jkbc,ci,abkj->ai'  , cc_obj.h2e.oovv, amp[0], amp[1])
    res +=     1.000000 * einsum('jkbc,cj,abki->ai'  , cc_obj.h2e.oovv, amp[0], amp[1])
    res +=    -1.000000 * einsum('Ijb,bi,aj,I->ai'   , cc_obj.h1p1e.ov, amp[0], amp[0], amp[2])
    res +=     1.000000 * einsum('jkbc,ci,aj,bk->ai' , cc_obj.h2e.oovv, amp[0], amp[0], amp[0])
    return res

def res_1(cc_obj, amp):
    """
    Generated by gen-cceqs.py at 2023-09-20 12:34:15
    Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    Hostname:     pauling001
    Git Branch:   pauling
    Git Commit:   323903bb20c46a422c99b2415498f89681509a89

    amp[0]: a+_v a_o
    amp[1]: a+_v a+_v a_o a_o
    amp[2]: b+
    amp[3]: b+ a+_v a_o
    amp[4]: b+ b+
    amp[5]: b+ b+ a+_v a_o
    res   :  a+_o a+_o a_v a_v

    ih = 0, ibra = 1, len(tmp.terms) = 1
    ih = 1, ibra = 1, len(tmp.terms) = 18
    ih = 2, ibra = 1, len(tmp.terms) = 53
    ih = 3, ibra = 1, len(tmp.terms) = 22
    ih = 4, ibra = 1, len(tmp.terms) = 1

    NOTE: the equation is not truncted.
    """
    res  =     1.000000 * einsum('baji->abij'        , cc_obj.h2e.vvoo)
    res +=     1.000000 * einsum('ki,bakj->abij'     , cc_obj.h1e.oo, amp[1])
    res +=    -1.000000 * einsum('kj,baki->abij'     , cc_obj.h1e.oo, amp[1])
    res +=     1.000000 * einsum('ac,bcji->abij'     , cc_obj.h1e.vv, amp[1])
    res +=    -1.000000 * einsum('bc,acji->abij'     , cc_obj.h1e.vv, amp[1])
    res +=     1.000000 * einsum('Iai,Ibj->abij'     , cc_obj.h1p1e.vo, amp[3])
    res +=    -1.000000 * einsum('Iaj,Ibi->abij'     , cc_obj.h1p1e.vo, amp[3])
    res +=    -1.000000 * einsum('Ibi,Iaj->abij'     , cc_obj.h1p1e.vo, amp[3])
    res +=     1.000000 * einsum('Ibj,Iai->abij'     , cc_obj.h1p1e.vo, amp[3])
    res +=    -1.000000 * einsum('kaji,bk->abij'     , cc_obj.h2e.ovoo, amp[0])
    res +=     1.000000 * einsum('kbji,ak->abij'     , cc_obj.h2e.ovoo, amp[0])
    res +=    -1.000000 * einsum('baic,cj->abij'     , cc_obj.h2e.vvov, amp[0])
    res +=     1.000000 * einsum('bajc,ci->abij'     , cc_obj.h2e.vvov, amp[0])
    res +=    -0.500000 * einsum('klji,balk->abij'   , cc_obj.h2e.oooo, amp[1])
    res +=     1.000000 * einsum('kaic,bckj->abij'   , cc_obj.h2e.ovov, amp[1])
    res +=    -1.000000 * einsum('kajc,bcki->abij'   , cc_obj.h2e.ovov, amp[1])
    res +=    -1.000000 * einsum('kbic,ackj->abij'   , cc_obj.h2e.ovov, amp[1])
    res +=     1.000000 * einsum('kbjc,acki->abij'   , cc_obj.h2e.ovov, amp[1])
    res +=    -0.500000 * einsum('bacd,dcji->abij'   , cc_obj.h2e.vvvv, amp[1])
    res +=    -1.000000 * einsum('kc,ak,bcji->abij'  , cc_obj.h1e.ov, amp[0], amp[1])
    res +=     1.000000 * einsum('kc,bk,acji->abij'  , cc_obj.h1e.ov, amp[0], amp[1])
    res +=     1.000000 * einsum('kc,ci,bakj->abij'  , cc_obj.h1e.ov, amp[0], amp[1])
    res +=    -1.000000 * einsum('kc,cj,baki->abij'  , cc_obj.h1e.ov, amp[0], amp[1])
    res +=    -1.000000 * einsum('Iki,ak,Ibj->abij'  , cc_obj.h1p1e.oo, amp[0], amp[3])
    res +=     1.000000 * einsum('Iki,bk,Iaj->abij'  , cc_obj.h1p1e.oo, amp[0], amp[3])
    res +=     1.000000 * einsum('Iki,bakj,I->abij'  , cc_obj.h1p1e.oo, amp[1], amp[2])
    res +=     1.000000 * einsum('Ikj,ak,Ibi->abij'  , cc_obj.h1p1e.oo, amp[0], amp[3])
    res +=    -1.000000 * einsum('Ikj,bk,Iai->abij'  , cc_obj.h1p1e.oo, amp[0], amp[3])
    res +=    -1.000000 * einsum('Ikj,baki,I->abij'  , cc_obj.h1p1e.oo, amp[1], amp[2])
    res +=     1.000000 * einsum('Iac,ci,Ibj->abij'  , cc_obj.h1p1e.vv, amp[0], amp[3])
    res +=    -1.000000 * einsum('Iac,cj,Ibi->abij'  , cc_obj.h1p1e.vv, amp[0], amp[3])
    res +=     1.000000 * einsum('Iac,bcji,I->abij'  , cc_obj.h1p1e.vv, amp[1], amp[2])
    res +=    -1.000000 * einsum('Ibc,ci,Iaj->abij'  , cc_obj.h1p1e.vv, amp[0], amp[3])
    res +=     1.000000 * einsum('Ibc,cj,Iai->abij'  , cc_obj.h1p1e.vv, amp[0], amp[3])
    res +=    -1.000000 * einsum('Ibc,acji,I->abij'  , cc_obj.h1p1e.vv, amp[1], amp[2])
    res +=     1.000000 * einsum('klji,bk,al->abij'  , cc_obj.h2e.oooo, amp[0], amp[0])
    res +=     1.000000 * einsum('kaic,cj,bk->abij'  , cc_obj.h2e.ovov, amp[0], amp[0])
    res +=    -1.000000 * einsum('kajc,ci,bk->abij'  , cc_obj.h2e.ovov, amp[0], amp[0])
    res +=    -1.000000 * einsum('kbic,cj,ak->abij'  , cc_obj.h2e.ovov, amp[0], amp[0])
    res +=     1.000000 * einsum('kbjc,ci,ak->abij'  , cc_obj.h2e.ovov, amp[0], amp[0])
    res +=     1.000000 * einsum('bacd,di,cj->abij'  , cc_obj.h2e.vvvv, amp[0], amp[0])
    res +=     1.000000 * einsum('Ikc,acji,Ibk->abij', cc_obj.h1p1e.ov, amp[1], amp[3])
    res +=    -1.000000 * einsum('Ikc,acki,Ibj->abij', cc_obj.h1p1e.ov, amp[1], amp[3])
    res +=     1.000000 * einsum('Ikc,ackj,Ibi->abij', cc_obj.h1p1e.ov, amp[1], amp[3])
    res +=    -1.000000 * einsum('Ikc,baki,Icj->abij', cc_obj.h1p1e.ov, amp[1], amp[3])
    res +=     1.000000 * einsum('Ikc,bakj,Ici->abij', cc_obj.h1p1e.ov, amp[1], amp[3])
    res +=    -1.000000 * einsum('Ikc,bcji,Iak->abij', cc_obj.h1p1e.ov, amp[1], amp[3])
    res +=     1.000000 * einsum('Ikc,bcki,Iaj->abij', cc_obj.h1p1e.ov, amp[1], amp[3])
    res +=    -1.000000 * einsum('Ikc,bckj,Iai->abij', cc_obj.h1p1e.ov, amp[1], amp[3])
    res +=     1.000000 * einsum('klic,ak,bclj->abij', cc_obj.h2e.ooov, amp[0], amp[1])
    res +=    -1.000000 * einsum('klic,bk,aclj->abij', cc_obj.h2e.ooov, amp[0], amp[1])
    res +=     0.500000 * einsum('klic,cj,balk->abij', cc_obj.h2e.ooov, amp[0], amp[1])
    res +=    -1.000000 * einsum('klic,ck,balj->abij', cc_obj.h2e.ooov, amp[0], amp[1])
    res +=    -1.000000 * einsum('kljc,ak,bcli->abij', cc_obj.h2e.ooov, amp[0], amp[1])
    res +=     1.000000 * einsum('kljc,bk,acli->abij', cc_obj.h2e.ooov, amp[0], amp[1])
    res +=    -0.500000 * einsum('kljc,ci,balk->abij', cc_obj.h2e.ooov, amp[0], amp[1])
    res +=     1.000000 * einsum('kljc,ck,bali->abij', cc_obj.h2e.ooov, amp[0], amp[1])
    res +=     0.500000 * einsum('kacd,bk,dcji->abij', cc_obj.h2e.ovvv, amp[0], amp[1])
    res +=    -1.000000 * einsum('kacd,di,bckj->abij', cc_obj.h2e.ovvv, amp[0], amp[1])
    res +=     1.000000 * einsum('kacd,dj,bcki->abij', cc_obj.h2e.ovvv, amp[0], amp[1])
    res +=    -1.000000 * einsum('kacd,dk,bcji->abij', cc_obj.h2e.ovvv, amp[0], amp[1])
    res +=    -0.500000 * einsum('kbcd,ak,dcji->abij', cc_obj.h2e.ovvv, amp[0], amp[1])
    res +=     1.000000 * einsum('kbcd,di,ackj->abij', cc_obj.h2e.ovvv, amp[0], amp[1])
    res +=    -1.000000 * einsum('kbcd,dj,acki->abij', cc_obj.h2e.ovvv, amp[0], amp[1])
    res +=     1.000000 * einsum('kbcd,dk,acji->abij', cc_obj.h2e.ovvv, amp[0], amp[1])
    res +=     0.500000 * einsum('klcd,adji,bclk->abij', cc_obj.h2e.oovv, amp[1], amp[1])
    res +=    -1.000000 * einsum('klcd,adki,bclj->abij', cc_obj.h2e.oovv, amp[1], amp[1])
    res +=    -0.500000 * einsum('klcd,baki,dclj->abij', cc_obj.h2e.oovv, amp[1], amp[1])
    res +=    -0.500000 * einsum('klcd,bdji,aclk->abij', cc_obj.h2e.oovv, amp[1], amp[1])
    res +=     1.000000 * einsum('klcd,bdki,aclj->abij', cc_obj.h2e.oovv, amp[1], amp[1])
    res +=     0.250000 * einsum('klcd,dcji,balk->abij', cc_obj.h2e.oovv, amp[1], amp[1])
    res +=    -0.500000 * einsum('klcd,dcki,balj->abij', cc_obj.h2e.oovv, amp[1], amp[1])
    res +=    -1.000000 * einsum('Ikc,ak,bcji,I->abij', cc_obj.h1p1e.ov, amp[0], amp[1], amp[2])
    res +=     1.000000 * einsum('Ikc,bk,acji,I->abij', cc_obj.h1p1e.ov, amp[0], amp[1], amp[2])
    res +=    -1.000000 * einsum('Ikc,ci,ak,Ibj->abij', cc_obj.h1p1e.ov, amp[0], amp[0], amp[3])
    res +=     1.000000 * einsum('Ikc,ci,bk,Iaj->abij', cc_obj.h1p1e.ov, amp[0], amp[0], amp[3])
    res +=     1.000000 * einsum('Ikc,ci,bakj,I->abij', cc_obj.h1p1e.ov, amp[0], amp[1], amp[2])
    res +=     1.000000 * einsum('Ikc,cj,ak,Ibi->abij', cc_obj.h1p1e.ov, amp[0], amp[0], amp[3])
    res +=    -1.000000 * einsum('Ikc,cj,bk,Iai->abij', cc_obj.h1p1e.ov, amp[0], amp[0], amp[3])
    res +=    -1.000000 * einsum('Ikc,cj,baki,I->abij', cc_obj.h1p1e.ov, amp[0], amp[1], amp[2])
    res +=    -1.000000 * einsum('klic,cj,bk,al->abij', cc_obj.h2e.ooov, amp[0], amp[0], amp[0])
    res +=     1.000000 * einsum('kljc,ci,bk,al->abij', cc_obj.h2e.ooov, amp[0], amp[0], amp[0])
    res +=    -1.000000 * einsum('kacd,di,cj,bk->abij', cc_obj.h2e.ovvv, amp[0], amp[0], amp[0])
    res +=     1.000000 * einsum('kbcd,di,cj,ak->abij', cc_obj.h2e.ovvv, amp[0], amp[0], amp[0])
    res +=    -1.000000 * einsum('klcd,ak,dl,bcji->abij', cc_obj.h2e.oovv, amp[0], amp[0], amp[1])
    res +=    -0.500000 * einsum('klcd,bk,al,dcji->abij', cc_obj.h2e.oovv, amp[0], amp[0], amp[1])
    res +=     1.000000 * einsum('klcd,bk,dl,acji->abij', cc_obj.h2e.oovv, amp[0], amp[0], amp[1])
    res +=    -1.000000 * einsum('klcd,di,ak,bclj->abij', cc_obj.h2e.oovv, amp[0], amp[0], amp[1])
    res +=     1.000000 * einsum('klcd,di,bk,aclj->abij', cc_obj.h2e.oovv, amp[0], amp[0], amp[1])
    res +=    -0.500000 * einsum('klcd,di,cj,balk->abij', cc_obj.h2e.oovv, amp[0], amp[0], amp[1])
    res +=     1.000000 * einsum('klcd,di,ck,balj->abij', cc_obj.h2e.oovv, amp[0], amp[0], amp[1])
    res +=     1.000000 * einsum('klcd,dj,ak,bcli->abij', cc_obj.h2e.oovv, amp[0], amp[0], amp[1])
    res +=    -1.000000 * einsum('klcd,dj,bk,acli->abij', cc_obj.h2e.oovv, amp[0], amp[0], amp[1])
    res +=    -1.000000 * einsum('klcd,dj,ck,bali->abij', cc_obj.h2e.oovv, amp[0], amp[0], amp[1])
    res +=     1.000000 * einsum('klcd,di,cj,bk,al->abij', cc_obj.h2e.oovv, amp[0], amp[0], amp[0], amp[0])
    return res

def res_2(cc_obj, amp):
    """
    Generated by gen-cceqs.py at 2023-09-20 12:48:28
    Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    Hostname:     pauling001
    Git Branch:   pauling
    Git Commit:   323903bb20c46a422c99b2415498f89681509a89

    amp[0]: a+_v a_o
    amp[1]: a+_v a+_v a_o a_o
    amp[2]: b+
    amp[3]: b+ a+_v a_o
    amp[4]: b+ b+
    amp[5]: b+ b+ a+_v a_o
    res   :  b

    ih = 0, ibra = 2, len(tmp.terms) = 1
    ih = 1, ibra = 2, len(tmp.terms) = 5
    ih = 2, ibra = 2, len(tmp.terms) = 3
    ih = 3, ibra = 2, len(tmp.terms) = 0
    """
    res  =     1.000000 * einsum('I->I'              , cc_obj.h1p_eff)
    res +=     1.000000 * einsum('J,IJ->I'           , cc_obj.h1p_eff, amp[4])
    res +=     1.000000 * einsum('IJ,J->I'           , cc_obj.h1p, amp[2])
    res +=     1.000000 * einsum('ia,Iai->I'         , cc_obj.h1e.ov, amp[3])
    res +=     1.000000 * einsum('Iia,ai->I'         , cc_obj.h1p1e.ov, amp[0])
    res +=     1.000000 * einsum('Jia,IJai->I'       , cc_obj.h1p1e.ov, amp[5])
    res +=     1.000000 * einsum('Jia,J,Iai->I'      , cc_obj.h1p1e.ov, amp[2], amp[3])
    res +=     1.000000 * einsum('Jia,ai,IJ->I'      , cc_obj.h1p1e.ov, amp[0], amp[4])
    res +=    -1.000000 * einsum('ijab,bi,Iaj->I'    , cc_obj.h2e.oovv, amp[0], amp[3])
    return res

def res_3(cc_obj, amp):
    """
    Generated by gen-cceqs.py at 2023-09-20 12:48:33
    Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    Hostname:     pauling001
    Git Branch:   pauling
    Git Commit:   323903bb20c46a422c99b2415498f89681509a89

    amp[0]: a+_v a_o
    amp[1]: a+_v a+_v a_o a_o
    amp[2]: b+
    amp[3]: b+ a+_v a_o
    amp[4]: b+ b+
    amp[5]: b+ b+ a+_v a_o
    res   :  b a+_o a_v

    ih = 0, ibra = 3, len(tmp.terms) = 1
    ih = 1, ibra = 3, len(tmp.terms) = 11
    ih = 2, ibra = 3, len(tmp.terms) = 21
    ih = 3, ibra = 3, len(tmp.terms) = 6
    ih = 4, ibra = 3, len(tmp.terms) = 0
    """
    res  =     1.000000 * einsum('Iai->Iai'          , cc_obj.h1p1e.vo)
    res +=     1.000000 * einsum('J,IJai->Iai'       , cc_obj.h1p_eff, amp[5])
    res +=    -1.000000 * einsum('ji,Iaj->Iai'       , cc_obj.h1e.oo, amp[3])
    res +=     1.000000 * einsum('ab,Ibi->Iai'       , cc_obj.h1e.vv, amp[3])
    res +=     1.000000 * einsum('IJ,Jai->Iai'       , cc_obj.h1p, amp[3])
    res +=    -1.000000 * einsum('Iji,aj->Iai'       , cc_obj.h1p1e.oo, amp[0])
    res +=     1.000000 * einsum('Iab,bi->Iai'       , cc_obj.h1p1e.vv, amp[0])
    res +=     1.000000 * einsum('Jai,IJ->Iai'       , cc_obj.h1p1e.vo, amp[4])
    res +=    -1.000000 * einsum('Ijb,abji->Iai'     , cc_obj.h1p1e.ov, amp[1])
    res +=    -1.000000 * einsum('Jji,IJaj->Iai'     , cc_obj.h1p1e.oo, amp[5])
    res +=     1.000000 * einsum('Jab,IJbi->Iai'     , cc_obj.h1p1e.vv, amp[5])
    res +=    -1.000000 * einsum('jaib,Ibj->Iai'     , cc_obj.h2e.ovov, amp[3])
    res +=    -1.000000 * einsum('jb,aj,Ibi->Iai'    , cc_obj.h1e.ov, amp[0], amp[3])
    res +=    -1.000000 * einsum('jb,bi,Iaj->Iai'    , cc_obj.h1e.ov, amp[0], amp[3])
    res +=    -1.000000 * einsum('Ijb,bi,aj->Iai'    , cc_obj.h1p1e.ov, amp[0], amp[0])
    res +=    -1.000000 * einsum('Jji,J,Iaj->Iai'    , cc_obj.h1p1e.oo, amp[2], amp[3])
    res +=    -1.000000 * einsum('Jji,aj,IJ->Iai'    , cc_obj.h1p1e.oo, amp[0], amp[4])
    res +=     1.000000 * einsum('Jab,J,Ibi->Iai'    , cc_obj.h1p1e.vv, amp[2], amp[3])
    res +=     1.000000 * einsum('Jab,bi,IJ->Iai'    , cc_obj.h1p1e.vv, amp[0], amp[4])
    res +=    -1.000000 * einsum('Jjb,aj,IJbi->Iai'  , cc_obj.h1p1e.ov, amp[0], amp[5])
    res +=    -1.000000 * einsum('Jjb,bi,IJaj->Iai'  , cc_obj.h1p1e.ov, amp[0], amp[5])
    res +=     1.000000 * einsum('Jjb,bj,IJai->Iai'  , cc_obj.h1p1e.ov, amp[0], amp[5])
    res +=    -1.000000 * einsum('Jjb,Iaj,Jbi->Iai'  , cc_obj.h1p1e.ov, amp[3], amp[3])
    res +=    -1.000000 * einsum('Jjb,Ibi,Jaj->Iai'  , cc_obj.h1p1e.ov, amp[3], amp[3])
    res +=     1.000000 * einsum('Jjb,Ibj,Jai->Iai'  , cc_obj.h1p1e.ov, amp[3], amp[3])
    res +=    -1.000000 * einsum('Jjb,abji,IJ->Iai'  , cc_obj.h1p1e.ov, amp[1], amp[4])
    res +=    -1.000000 * einsum('jkib,aj,Ibk->Iai'  , cc_obj.h2e.ooov, amp[0], amp[3])
    res +=     1.000000 * einsum('jkib,bj,Iak->Iai'  , cc_obj.h2e.ooov, amp[0], amp[3])
    res +=     1.000000 * einsum('jabc,ci,Ibj->Iai'  , cc_obj.h2e.ovvv, amp[0], amp[3])
    res +=    -1.000000 * einsum('jabc,cj,Ibi->Iai'  , cc_obj.h2e.ovvv, amp[0], amp[3])
    res +=     1.000000 * einsum('jkbc,acji,Ibk->Iai', cc_obj.h2e.oovv, amp[1], amp[3])
    res +=     0.500000 * einsum('jkbc,ackj,Ibi->Iai', cc_obj.h2e.oovv, amp[1], amp[3])
    res +=     0.500000 * einsum('jkbc,cbji,Iak->Iai', cc_obj.h2e.oovv, amp[1], amp[3])
    res +=    -1.000000 * einsum('Jjb,aj,J,Ibi->Iai' , cc_obj.h1p1e.ov, amp[0], amp[2], amp[3])
    res +=    -1.000000 * einsum('Jjb,bi,J,Iaj->Iai' , cc_obj.h1p1e.ov, amp[0], amp[2], amp[3])
    res +=    -1.000000 * einsum('Jjb,bi,aj,IJ->Iai' , cc_obj.h1p1e.ov, amp[0], amp[0], amp[4])
    res +=    -1.000000 * einsum('jkbc,aj,ck,Ibi->Iai', cc_obj.h2e.oovv, amp[0], amp[0], amp[3])
    res +=     1.000000 * einsum('jkbc,ci,aj,Ibk->Iai', cc_obj.h2e.oovv, amp[0], amp[0], amp[3])
    res +=    -1.000000 * einsum('jkbc,ci,bj,Iak->Iai', cc_obj.h2e.oovv, amp[0], amp[0], amp[3])
    return res

def res_4(cc_obj, amp):
    """
    Generated by gen-cceqs.py at 2023-09-20 12:53:02
    Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    Hostname:     pauling001
    Git Branch:   pauling
    Git Commit:   323903bb20c46a422c99b2415498f89681509a89

    amp[0]: a+_v a_o
    amp[1]: a+_v a+_v a_o a_o
    amp[2]: b+
    amp[3]: b+ a+_v a_o
    amp[4]: b+ b+
    amp[5]: b+ b+ a+_v a_o
    res   :  b b

    ih = 0, ibra = 4, len(tmp.terms) = 0
    ih = 1, ibra = 4, len(tmp.terms) = 5
    ih = 2, ibra = 4, len(tmp.terms) = 5
    ih = 3, ibra = 4, len(tmp.terms) = 0
    """
    res  =     1.000000 * einsum('IK,JK->IJ'         , cc_obj.h1p, amp[4])
    res +=     1.000000 * einsum('JK,IK->IJ'         , cc_obj.h1p, amp[4])
    res +=     1.000000 * einsum('ia,IJai->IJ'       , cc_obj.h1e.ov, amp[5])
    res +=     1.000000 * einsum('Iia,Jai->IJ'       , cc_obj.h1p1e.ov, amp[3])
    res +=     1.000000 * einsum('Jia,Iai->IJ'       , cc_obj.h1p1e.ov, amp[3])
    res +=     1.000000 * einsum('Kia,K,IJai->IJ'    , cc_obj.h1p1e.ov, amp[2], amp[5])
    res +=     1.000000 * einsum('Kia,Iai,JK->IJ'    , cc_obj.h1p1e.ov, amp[3], amp[4])
    res +=     1.000000 * einsum('Kia,Jai,IK->IJ'    , cc_obj.h1p1e.ov, amp[3], amp[4])
    res +=    -1.000000 * einsum('ijab,bi,IJaj->IJ'  , cc_obj.h2e.oovv, amp[0], amp[5])
    res +=    -1.000000 * einsum('ijab,Ibi,Jaj->IJ'  , cc_obj.h2e.oovv, amp[3], amp[3])
    return res

def res_5(cc_obj, amp):
    """
    Generated by gen-cceqs.py at 2023-09-20 12:53:09
    Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    Hostname:     pauling001
    Git Branch:   pauling
    Git Commit:   323903bb20c46a422c99b2415498f89681509a89

    amp[0]: a+_v a_o
    amp[1]: a+_v a+_v a_o a_o
    amp[2]: b+
    amp[3]: b+ a+_v a_o
    amp[4]: b+ b+
    amp[5]: b+ b+ a+_v a_o
    res   :  b b a+_o a_v

    ih = 0, ibra = 5, len(tmp.terms) = 0
    ih = 1, ibra = 5, len(tmp.terms) = 9
    ih = 2, ibra = 5, len(tmp.terms) = 34
    ih = 3, ibra = 5, len(tmp.terms) = 17
    ih = 4, ibra = 5, len(tmp.terms) = 0
    """
    res  =    -1.000000 * einsum('ji,IJaj->IJai'     , cc_obj.h1e.oo, amp[5])
    res +=     1.000000 * einsum('ab,IJbi->IJai'     , cc_obj.h1e.vv, amp[5])
    res +=     1.000000 * einsum('IK,JKai->IJai'     , cc_obj.h1p, amp[5])
    res +=     1.000000 * einsum('JK,IKai->IJai'     , cc_obj.h1p, amp[5])
    res +=    -1.000000 * einsum('Iji,Jaj->IJai'     , cc_obj.h1p1e.oo, amp[3])
    res +=     1.000000 * einsum('Iab,Jbi->IJai'     , cc_obj.h1p1e.vv, amp[3])
    res +=    -1.000000 * einsum('Jji,Iaj->IJai'     , cc_obj.h1p1e.oo, amp[3])
    res +=     1.000000 * einsum('Jab,Ibi->IJai'     , cc_obj.h1p1e.vv, amp[3])
    res +=    -1.000000 * einsum('jaib,IJbj->IJai'   , cc_obj.h2e.ovov, amp[5])
    res +=    -1.000000 * einsum('jb,aj,IJbi->IJai'  , cc_obj.h1e.ov, amp[0], amp[5])
    res +=    -1.000000 * einsum('jb,bi,IJaj->IJai'  , cc_obj.h1e.ov, amp[0], amp[5])
    res +=    -1.000000 * einsum('jb,Iaj,Jbi->IJai'  , cc_obj.h1e.ov, amp[3], amp[3])
    res +=    -1.000000 * einsum('jb,Ibi,Jaj->IJai'  , cc_obj.h1e.ov, amp[3], amp[3])
    res +=    -1.000000 * einsum('Ijb,aj,Jbi->IJai'  , cc_obj.h1p1e.ov, amp[0], amp[3])
    res +=    -1.000000 * einsum('Ijb,bi,Jaj->IJai'  , cc_obj.h1p1e.ov, amp[0], amp[3])
    res +=    -1.000000 * einsum('Jjb,aj,Ibi->IJai'  , cc_obj.h1p1e.ov, amp[0], amp[3])
    res +=    -1.000000 * einsum('Jjb,bi,Iaj->IJai'  , cc_obj.h1p1e.ov, amp[0], amp[3])
    res +=    -1.000000 * einsum('Kji,K,IJaj->IJai'  , cc_obj.h1p1e.oo, amp[2], amp[5])
    res +=    -1.000000 * einsum('Kji,Iaj,JK->IJai'  , cc_obj.h1p1e.oo, amp[3], amp[4])
    res +=    -1.000000 * einsum('Kji,Jaj,IK->IJai'  , cc_obj.h1p1e.oo, amp[3], amp[4])
    res +=     1.000000 * einsum('Kab,K,IJbi->IJai'  , cc_obj.h1p1e.vv, amp[2], amp[5])
    res +=     1.000000 * einsum('Kab,Ibi,JK->IJai'  , cc_obj.h1p1e.vv, amp[3], amp[4])
    res +=     1.000000 * einsum('Kab,Jbi,IK->IJai'  , cc_obj.h1p1e.vv, amp[3], amp[4])
    res +=    -1.000000 * einsum('Kjb,Iaj,JKbi->IJai', cc_obj.h1p1e.ov, amp[3], amp[5])
    res +=    -1.000000 * einsum('Kjb,Ibi,JKaj->IJai', cc_obj.h1p1e.ov, amp[3], amp[5])
    res +=     1.000000 * einsum('Kjb,Ibj,JKai->IJai', cc_obj.h1p1e.ov, amp[3], amp[5])
    res +=    -1.000000 * einsum('Kjb,Jaj,IKbi->IJai', cc_obj.h1p1e.ov, amp[3], amp[5])
    res +=    -1.000000 * einsum('Kjb,Jbi,IKaj->IJai', cc_obj.h1p1e.ov, amp[3], amp[5])
    res +=     1.000000 * einsum('Kjb,Jbj,IKai->IJai', cc_obj.h1p1e.ov, amp[3], amp[5])
    res +=     1.000000 * einsum('Kjb,Kai,IJbj->IJai', cc_obj.h1p1e.ov, amp[3], amp[5])
    res +=    -1.000000 * einsum('Kjb,Kaj,IJbi->IJai', cc_obj.h1p1e.ov, amp[3], amp[5])
    res +=    -1.000000 * einsum('Kjb,Kbi,IJaj->IJai', cc_obj.h1p1e.ov, amp[3], amp[5])
    res +=    -1.000000 * einsum('jkib,aj,IJbk->IJai', cc_obj.h2e.ooov, amp[0], amp[5])
    res +=     1.000000 * einsum('jkib,bj,IJak->IJai', cc_obj.h2e.ooov, amp[0], amp[5])
    res +=    -1.000000 * einsum('jkib,Iaj,Jbk->IJai', cc_obj.h2e.ooov, amp[3], amp[3])
    res +=     1.000000 * einsum('jkib,Ibj,Jak->IJai', cc_obj.h2e.ooov, amp[3], amp[3])
    res +=     1.000000 * einsum('jabc,ci,IJbj->IJai', cc_obj.h2e.ovvv, amp[0], amp[5])
    res +=    -1.000000 * einsum('jabc,cj,IJbi->IJai', cc_obj.h2e.ovvv, amp[0], amp[5])
    res +=     1.000000 * einsum('jabc,Ici,Jbj->IJai', cc_obj.h2e.ovvv, amp[3], amp[3])
    res +=    -1.000000 * einsum('jabc,Icj,Jbi->IJai', cc_obj.h2e.ovvv, amp[3], amp[3])
    res +=     1.000000 * einsum('jkbc,acji,IJbk->IJai', cc_obj.h2e.oovv, amp[1], amp[5])
    res +=     0.500000 * einsum('jkbc,ackj,IJbi->IJai', cc_obj.h2e.oovv, amp[1], amp[5])
    res +=     0.500000 * einsum('jkbc,cbji,IJak->IJai', cc_obj.h2e.oovv, amp[1], amp[5])
    res +=    -1.000000 * einsum('Kjb,K,Iaj,Jbi->IJai', cc_obj.h1p1e.ov, amp[2], amp[3], amp[3])
    res +=    -1.000000 * einsum('Kjb,K,Ibi,Jaj->IJai', cc_obj.h1p1e.ov, amp[2], amp[3], amp[3])
    res +=    -1.000000 * einsum('Kjb,aj,K,IJbi->IJai', cc_obj.h1p1e.ov, amp[0], amp[2], amp[5])
    res +=    -1.000000 * einsum('Kjb,aj,Ibi,JK->IJai', cc_obj.h1p1e.ov, amp[0], amp[3], amp[4])
    res +=    -1.000000 * einsum('Kjb,aj,Jbi,IK->IJai', cc_obj.h1p1e.ov, amp[0], amp[3], amp[4])
    res +=    -1.000000 * einsum('Kjb,bi,K,IJaj->IJai', cc_obj.h1p1e.ov, amp[0], amp[2], amp[5])
    res +=    -1.000000 * einsum('Kjb,bi,Iaj,JK->IJai', cc_obj.h1p1e.ov, amp[0], amp[3], amp[4])
    res +=    -1.000000 * einsum('Kjb,bi,Jaj,IK->IJai', cc_obj.h1p1e.ov, amp[0], amp[3], amp[4])
    res +=    -1.000000 * einsum('jkbc,aj,ck,IJbi->IJai', cc_obj.h2e.oovv, amp[0], amp[0], amp[5])
    res +=     1.000000 * einsum('jkbc,aj,Ici,Jbk->IJai', cc_obj.h2e.oovv, amp[0], amp[3], amp[3])
    res +=    -1.000000 * einsum('jkbc,aj,Ick,Jbi->IJai', cc_obj.h2e.oovv, amp[0], amp[3], amp[3])
    res +=     1.000000 * einsum('jkbc,ci,aj,IJbk->IJai', cc_obj.h2e.oovv, amp[0], amp[0], amp[5])
    res +=    -1.000000 * einsum('jkbc,ci,bj,IJak->IJai', cc_obj.h2e.oovv, amp[0], amp[0], amp[5])
    res +=     1.000000 * einsum('jkbc,ci,Iaj,Jbk->IJai', cc_obj.h2e.oovv, amp[0], amp[3], amp[3])
    res +=    -1.000000 * einsum('jkbc,ci,Ibj,Jak->IJai', cc_obj.h2e.oovv, amp[0], amp[3], amp[3])
    res +=     1.000000 * einsum('jkbc,cj,Iak,Jbi->IJai', cc_obj.h2e.oovv, amp[0], amp[3], amp[3])
    res +=     1.000000 * einsum('jkbc,cj,Ibi,Jak->IJai', cc_obj.h2e.oovv, amp[0], amp[3], amp[3])
    return res

def ene(cc_obj, amp):
    """
    Generated by gen-cceqs.py at 2023-09-20 13:06:15
    Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    Hostname:     pauling001
    Git Branch:   pauling
    Git Commit:   323903bb20c46a422c99b2415498f89681509a89

    amp[0]: a+_v a_o
    amp[1]: a+_v a+_v a_o a_o
    amp[2]: b+
    amp[3]: b+ a+_v a_o
    amp[4]: b+ b+
    amp[5]: b+ b+ a+_v a_o

    ih = 0, ibra = 6, len(tmp.terms) = 0
    ih = 1, ibra = 6, len(tmp.terms) = 4
    ih = 2, ibra = 6, len(tmp.terms) = 2
    ih = 3, ibra = 6, len(tmp.terms) = 0
    """
    res  =     1.000000 * einsum('I,I->'             , cc_obj.h1p_eff, amp[2])
    res +=     1.000000 * einsum('ia,ai->'           , cc_obj.h1e.ov, amp[0])
    res +=     1.000000 * einsum('Iia,Iai->'         , cc_obj.h1p1e.ov, amp[3])
    res +=     0.250000 * einsum('ijab,baji->'       , cc_obj.h2e.oovv, amp[1])
    res +=     1.000000 * einsum('Iia,ai,I->'        , cc_obj.h1p1e.ov, amp[0], amp[2])
    res +=    -0.500000 * einsum('ijab,bi,aj->'      , cc_obj.h2e.oovv, amp[0], amp[0])
    return res
