import numpy, functools
einsum = functools.partial(numpy.einsum, optimize=True)

def res_0(cc_obj, amp):
    """
    Generated by gen-cceqs.py at 2023-09-19 15:40:41
    Machine Info: Darwin 23.0.0
    Hostname:     Junjies-MacBook-Air.local
    Git Branch:   pauling
    Git Commit:   abc0d69e1b9e4f80a5539b87a023d6b3e073385f

    amp[0]: a+_v a_o
    amp[1]: a+_v a+_v a_o a_o
    amp[2]: b+
    amp[3]: b+ a+_v a_o
    res   :  a+_o a_v

    ih = 0, ibra = 0, len(tmp.terms) = 1
    ih = 1, ibra = 0, len(tmp.terms) = 10
    ih = 2, ibra = 0, len(tmp.terms) = 12
    ih = 3, ibra = 0, len(tmp.terms) = 2
    ih = 4, ibra = 0, len(tmp.terms) = 0
    """
    res  =     1.000000 * einsum('ai->ai'            , cc_obj.h1e)
    res +=     1.000000 * einsum('I,Iai->ai'         , cc_obj.h1p_eff, amp[3])
    res +=    -1.000000 * einsum('ji,aj->ai'         , cc_obj.h1e, amp[0])
    res +=     1.000000 * einsum('ab,bi->ai'         , cc_obj.h1e, amp[0])
    res +=     1.000000 * einsum('Iai,I->ai'         , cc_obj.h1p1e, amp[2])
    res +=    -1.000000 * einsum('jb,abji->ai'       , cc_obj.h1e, amp[1])
    res +=    -1.000000 * einsum('Iji,Iaj->ai'       , cc_obj.h1p1e, amp[3])
    res +=     1.000000 * einsum('Iab,Ibi->ai'       , cc_obj.h1p1e, amp[3])
    res +=    -1.000000 * einsum('jaib,bj->ai'       , cc_obj.h2e, amp[0])
    res +=     0.500000 * einsum('jkib,abkj->ai'     , cc_obj.h2e, amp[1])
    res +=    -0.500000 * einsum('jabc,cbji->ai'     , cc_obj.h2e, amp[1])
    res +=    -1.000000 * einsum('jb,bi,aj->ai'      , cc_obj.h1e, amp[0], amp[0])
    res +=    -1.000000 * einsum('Iji,aj,I->ai'      , cc_obj.h1p1e, amp[0], amp[2])
    res +=     1.000000 * einsum('Iab,bi,I->ai'      , cc_obj.h1p1e, amp[0], amp[2])
    res +=    -1.000000 * einsum('Ijb,aj,Ibi->ai'    , cc_obj.h1p1e, amp[0], amp[3])
    res +=    -1.000000 * einsum('Ijb,bi,Iaj->ai'    , cc_obj.h1p1e, amp[0], amp[3])
    res +=     1.000000 * einsum('Ijb,bj,Iai->ai'    , cc_obj.h1p1e, amp[0], amp[3])
    res +=    -1.000000 * einsum('Ijb,abji,I->ai'    , cc_obj.h1p1e, amp[1], amp[2])
    res +=    -1.000000 * einsum('jkib,aj,bk->ai'    , cc_obj.h2e, amp[0], amp[0])
    res +=     1.000000 * einsum('jabc,ci,bj->ai'    , cc_obj.h2e, amp[0], amp[0])
    res +=    -0.500000 * einsum('jkbc,aj,cbki->ai'  , cc_obj.h2e, amp[0], amp[1])
    res +=    -0.500000 * einsum('jkbc,ci,abkj->ai'  , cc_obj.h2e, amp[0], amp[1])
    res +=     1.000000 * einsum('jkbc,cj,abki->ai'  , cc_obj.h2e, amp[0], amp[1])
    res +=    -0.500000 * einsum('Ijb,bi,aj,I->ai'   , cc_obj.h1p1e, amp[0], amp[0], amp[2])
    res +=     0.500000 * einsum('jkbc,ci,aj,bk->ai' , cc_obj.h2e, amp[0], amp[0], amp[0])
    return res

def res_1(cc_obj, amp):
    """
    Generated by gen-cceqs.py at 2023-09-19 15:41:22
    Machine Info: Darwin 23.0.0
    Hostname:     Junjies-MacBook-Air.local
    Git Branch:   pauling
    Git Commit:   abc0d69e1b9e4f80a5539b87a023d6b3e073385f

    amp[0]: a+_v a_o
    amp[1]: a+_v a+_v a_o a_o
    amp[2]: b+
    amp[3]: b+ a+_v a_o
    res   :  a+_o a+_o a_v a_v

    ih = 0, ibra = 1, len(tmp.terms) = 1
    ih = 1, ibra = 1, len(tmp.terms) = 18
    ih = 2, ibra = 1, len(tmp.terms) = 53
    ih = 3, ibra = 1, len(tmp.terms) = 22
    ih = 4, ibra = 1, len(tmp.terms) = 1

    NOTE: the equation is not truncted.
    """
    res  =     1.000000 * einsum('baji->abij'        , cc_obj.h2e)
    res +=     1.000000 * einsum('ki,bakj->abij'     , cc_obj.h1e, amp[1])
    res +=    -1.000000 * einsum('kj,baki->abij'     , cc_obj.h1e, amp[1])
    res +=     1.000000 * einsum('ac,bcji->abij'     , cc_obj.h1e, amp[1])
    res +=    -1.000000 * einsum('bc,acji->abij'     , cc_obj.h1e, amp[1])
    res +=     1.000000 * einsum('Iai,Ibj->abij'     , cc_obj.h1p1e, amp[3])
    res +=    -1.000000 * einsum('Iaj,Ibi->abij'     , cc_obj.h1p1e, amp[3])
    res +=    -1.000000 * einsum('Ibi,Iaj->abij'     , cc_obj.h1p1e, amp[3])
    res +=     1.000000 * einsum('Ibj,Iai->abij'     , cc_obj.h1p1e, amp[3])
    res +=    -1.000000 * einsum('kaji,bk->abij'     , cc_obj.h2e, amp[0])
    res +=     1.000000 * einsum('kbji,ak->abij'     , cc_obj.h2e, amp[0])
    res +=    -1.000000 * einsum('baic,cj->abij'     , cc_obj.h2e, amp[0])
    res +=     1.000000 * einsum('bajc,ci->abij'     , cc_obj.h2e, amp[0])
    res +=    -0.500000 * einsum('klji,balk->abij'   , cc_obj.h2e, amp[1])
    res +=     1.000000 * einsum('kaic,bckj->abij'   , cc_obj.h2e, amp[1])
    res +=    -1.000000 * einsum('kajc,bcki->abij'   , cc_obj.h2e, amp[1])
    res +=    -1.000000 * einsum('kbic,ackj->abij'   , cc_obj.h2e, amp[1])
    res +=     1.000000 * einsum('kbjc,acki->abij'   , cc_obj.h2e, amp[1])
    res +=    -0.500000 * einsum('bacd,dcji->abij'   , cc_obj.h2e, amp[1])
    res +=    -1.000000 * einsum('kc,ak,bcji->abij'  , cc_obj.h1e, amp[0], amp[1])
    res +=     1.000000 * einsum('kc,bk,acji->abij'  , cc_obj.h1e, amp[0], amp[1])
    res +=     1.000000 * einsum('kc,ci,bakj->abij'  , cc_obj.h1e, amp[0], amp[1])
    res +=    -1.000000 * einsum('kc,cj,baki->abij'  , cc_obj.h1e, amp[0], amp[1])
    res +=    -1.000000 * einsum('Iki,ak,Ibj->abij'  , cc_obj.h1p1e, amp[0], amp[3])
    res +=     1.000000 * einsum('Iki,bk,Iaj->abij'  , cc_obj.h1p1e, amp[0], amp[3])
    res +=     1.000000 * einsum('Iki,bakj,I->abij'  , cc_obj.h1p1e, amp[1], amp[2])
    res +=     1.000000 * einsum('Ikj,ak,Ibi->abij'  , cc_obj.h1p1e, amp[0], amp[3])
    res +=    -1.000000 * einsum('Ikj,bk,Iai->abij'  , cc_obj.h1p1e, amp[0], amp[3])
    res +=    -1.000000 * einsum('Ikj,baki,I->abij'  , cc_obj.h1p1e, amp[1], amp[2])
    res +=     1.000000 * einsum('Iac,ci,Ibj->abij'  , cc_obj.h1p1e, amp[0], amp[3])
    res +=    -1.000000 * einsum('Iac,cj,Ibi->abij'  , cc_obj.h1p1e, amp[0], amp[3])
    res +=     1.000000 * einsum('Iac,bcji,I->abij'  , cc_obj.h1p1e, amp[1], amp[2])
    res +=    -1.000000 * einsum('Ibc,ci,Iaj->abij'  , cc_obj.h1p1e, amp[0], amp[3])
    res +=     1.000000 * einsum('Ibc,cj,Iai->abij'  , cc_obj.h1p1e, amp[0], amp[3])
    res +=    -1.000000 * einsum('Ibc,acji,I->abij'  , cc_obj.h1p1e, amp[1], amp[2])
    res +=     1.000000 * einsum('klji,bk,al->abij'  , cc_obj.h2e, amp[0], amp[0])
    res +=     1.000000 * einsum('kaic,cj,bk->abij'  , cc_obj.h2e, amp[0], amp[0])
    res +=    -1.000000 * einsum('kajc,ci,bk->abij'  , cc_obj.h2e, amp[0], amp[0])
    res +=    -1.000000 * einsum('kbic,cj,ak->abij'  , cc_obj.h2e, amp[0], amp[0])
    res +=     1.000000 * einsum('kbjc,ci,ak->abij'  , cc_obj.h2e, amp[0], amp[0])
    res +=     1.000000 * einsum('bacd,di,cj->abij'  , cc_obj.h2e, amp[0], amp[0])
    res +=     1.000000 * einsum('Ikc,acji,Ibk->abij', cc_obj.h1p1e, amp[1], amp[3])
    res +=    -1.000000 * einsum('Ikc,acki,Ibj->abij', cc_obj.h1p1e, amp[1], amp[3])
    res +=     1.000000 * einsum('Ikc,ackj,Ibi->abij', cc_obj.h1p1e, amp[1], amp[3])
    res +=    -1.000000 * einsum('Ikc,baki,Icj->abij', cc_obj.h1p1e, amp[1], amp[3])
    res +=     1.000000 * einsum('Ikc,bakj,Ici->abij', cc_obj.h1p1e, amp[1], amp[3])
    res +=    -1.000000 * einsum('Ikc,bcji,Iak->abij', cc_obj.h1p1e, amp[1], amp[3])
    res +=     1.000000 * einsum('Ikc,bcki,Iaj->abij', cc_obj.h1p1e, amp[1], amp[3])
    res +=    -1.000000 * einsum('Ikc,bckj,Iai->abij', cc_obj.h1p1e, amp[1], amp[3])
    res +=     1.000000 * einsum('klic,ak,bclj->abij', cc_obj.h2e, amp[0], amp[1])
    res +=    -1.000000 * einsum('klic,bk,aclj->abij', cc_obj.h2e, amp[0], amp[1])
    res +=     0.500000 * einsum('klic,cj,balk->abij', cc_obj.h2e, amp[0], amp[1])
    res +=    -1.000000 * einsum('klic,ck,balj->abij', cc_obj.h2e, amp[0], amp[1])
    res +=    -1.000000 * einsum('kljc,ak,bcli->abij', cc_obj.h2e, amp[0], amp[1])
    res +=     1.000000 * einsum('kljc,bk,acli->abij', cc_obj.h2e, amp[0], amp[1])
    res +=    -0.500000 * einsum('kljc,ci,balk->abij', cc_obj.h2e, amp[0], amp[1])
    res +=     1.000000 * einsum('kljc,ck,bali->abij', cc_obj.h2e, amp[0], amp[1])
    res +=     0.500000 * einsum('kacd,bk,dcji->abij', cc_obj.h2e, amp[0], amp[1])
    res +=    -1.000000 * einsum('kacd,di,bckj->abij', cc_obj.h2e, amp[0], amp[1])
    res +=     1.000000 * einsum('kacd,dj,bcki->abij', cc_obj.h2e, amp[0], amp[1])
    res +=    -1.000000 * einsum('kacd,dk,bcji->abij', cc_obj.h2e, amp[0], amp[1])
    res +=    -0.500000 * einsum('kbcd,ak,dcji->abij', cc_obj.h2e, amp[0], amp[1])
    res +=     1.000000 * einsum('kbcd,di,ackj->abij', cc_obj.h2e, amp[0], amp[1])
    res +=    -1.000000 * einsum('kbcd,dj,acki->abij', cc_obj.h2e, amp[0], amp[1])
    res +=     1.000000 * einsum('kbcd,dk,acji->abij', cc_obj.h2e, amp[0], amp[1])
    res +=     0.500000 * einsum('klcd,adji,bclk->abij', cc_obj.h2e, amp[1], amp[1])
    res +=    -1.000000 * einsum('klcd,adki,bclj->abij', cc_obj.h2e, amp[1], amp[1])
    res +=    -0.500000 * einsum('klcd,baki,dclj->abij', cc_obj.h2e, amp[1], amp[1])
    res +=    -0.500000 * einsum('klcd,bdji,aclk->abij', cc_obj.h2e, amp[1], amp[1])
    res +=     1.000000 * einsum('klcd,bdki,aclj->abij', cc_obj.h2e, amp[1], amp[1])
    res +=     0.250000 * einsum('klcd,dcji,balk->abij', cc_obj.h2e, amp[1], amp[1])
    res +=    -0.500000 * einsum('klcd,dcki,balj->abij', cc_obj.h2e, amp[1], amp[1])
    res +=    -0.500000 * einsum('Ikc,ak,bcji,I->abij', cc_obj.h1p1e, amp[0], amp[1], amp[2])
    res +=     0.500000 * einsum('Ikc,bk,acji,I->abij', cc_obj.h1p1e, amp[0], amp[1], amp[2])
    res +=    -0.500000 * einsum('Ikc,ci,ak,Ibj->abij', cc_obj.h1p1e, amp[0], amp[0], amp[3])
    res +=     0.500000 * einsum('Ikc,ci,bk,Iaj->abij', cc_obj.h1p1e, amp[0], amp[0], amp[3])
    res +=     0.500000 * einsum('Ikc,ci,bakj,I->abij', cc_obj.h1p1e, amp[0], amp[1], amp[2])
    res +=     0.500000 * einsum('Ikc,cj,ak,Ibi->abij', cc_obj.h1p1e, amp[0], amp[0], amp[3])
    res +=    -0.500000 * einsum('Ikc,cj,bk,Iai->abij', cc_obj.h1p1e, amp[0], amp[0], amp[3])
    res +=    -0.500000 * einsum('Ikc,cj,baki,I->abij', cc_obj.h1p1e, amp[0], amp[1], amp[2])
    res +=    -0.500000 * einsum('klic,cj,bk,al->abij', cc_obj.h2e, amp[0], amp[0], amp[0])
    res +=     0.500000 * einsum('kljc,ci,bk,al->abij', cc_obj.h2e, amp[0], amp[0], amp[0])
    res +=    -0.500000 * einsum('kacd,di,cj,bk->abij', cc_obj.h2e, amp[0], amp[0], amp[0])
    res +=     0.500000 * einsum('kbcd,di,cj,ak->abij', cc_obj.h2e, amp[0], amp[0], amp[0])
    res +=    -0.500000 * einsum('klcd,ak,dl,bcji->abij', cc_obj.h2e, amp[0], amp[0], amp[1])
    res +=    -0.250000 * einsum('klcd,bk,al,dcji->abij', cc_obj.h2e, amp[0], amp[0], amp[1])
    res +=     0.500000 * einsum('klcd,bk,dl,acji->abij', cc_obj.h2e, amp[0], amp[0], amp[1])
    res +=    -0.500000 * einsum('klcd,di,ak,bclj->abij', cc_obj.h2e, amp[0], amp[0], amp[1])
    res +=     0.500000 * einsum('klcd,di,bk,aclj->abij', cc_obj.h2e, amp[0], amp[0], amp[1])
    res +=    -0.250000 * einsum('klcd,di,cj,balk->abij', cc_obj.h2e, amp[0], amp[0], amp[1])
    res +=     0.500000 * einsum('klcd,di,ck,balj->abij', cc_obj.h2e, amp[0], amp[0], amp[1])
    res +=     0.500000 * einsum('klcd,dj,ak,bcli->abij', cc_obj.h2e, amp[0], amp[0], amp[1])
    res +=    -0.500000 * einsum('klcd,dj,bk,acli->abij', cc_obj.h2e, amp[0], amp[0], amp[1])
    res +=    -0.500000 * einsum('klcd,dj,ck,bali->abij', cc_obj.h2e, amp[0], amp[0], amp[1])
    res +=     0.083333 * einsum('klcd,di,cj,bk,al->abij', cc_obj.h2e, amp[0], amp[0], amp[0], amp[0])
    return res

def res_2(cc_obj, amp):
    """
    Generated by gen-cceqs.py at 2023-09-19 15:45:11
    Machine Info: Darwin 23.0.0
    Hostname:     Junjies-MacBook-Air.local
    Git Branch:   pauling
    Git Commit:   abc0d69e1b9e4f80a5539b87a023d6b3e073385f

    amp[0]: a+_v a_o
    amp[1]: a+_v a+_v a_o a_o
    amp[2]: b+
    amp[3]: b+ a+_v a_o
    res   :  b

    ih = 0, ibra = 2, len(tmp.terms) = 1
    ih = 1, ibra = 2, len(tmp.terms) = 3
    ih = 2, ibra = 2, len(tmp.terms) = 2
    ih = 3, ibra = 2, len(tmp.terms) = 0
    """
    res  =     1.000000 * einsum('I->I'              , cc_obj.h1p_eff)
    res +=     1.000000 * einsum('IJ,J->I'           , cc_obj.h1p, amp[2])
    res +=     1.000000 * einsum('ia,Iai->I'         , cc_obj.h1e, amp[3])
    res +=     1.000000 * einsum('Iia,ai->I'         , cc_obj.h1p1e, amp[0])
    res +=     1.000000 * einsum('Jia,J,Iai->I'      , cc_obj.h1p1e, amp[2], amp[3])
    res +=    -1.000000 * einsum('ijab,bi,Iaj->I'    , cc_obj.h2e, amp[0], amp[3])
    return res

def res_3(cc_obj, amp):
    """
    Generated by gen-cceqs.py at 2023-09-19 15:45:28
    Machine Info: Darwin 23.0.0
    Hostname:     Junjies-MacBook-Air.local
    Git Branch:   pauling
    Git Commit:   abc0d69e1b9e4f80a5539b87a023d6b3e073385f

    amp[0]: a+_v a_o
    amp[1]: a+_v a+_v a_o a_o
    amp[2]: b+
    amp[3]: b+ a+_v a_o
    res   :  b a+_o a_v

    ih = 0, ibra = 3, len(tmp.terms) = 1
    ih = 1, ibra = 3, len(tmp.terms) = 7
    ih = 2, ibra = 3, len(tmp.terms) = 15
    ih = 3, ibra = 3, len(tmp.terms) = 5
    ih = 4, ibra = 3, len(tmp.terms) = 0
    """
    res  =     1.000000 * einsum('Iai->Iai'          , cc_obj.h1p1e)
    res +=    -1.000000 * einsum('ji,Iaj->Iai'       , cc_obj.h1e, amp[3])
    res +=     1.000000 * einsum('ab,Ibi->Iai'       , cc_obj.h1e, amp[3])
    res +=     1.000000 * einsum('IJ,Jai->Iai'       , cc_obj.h1p, amp[3])
    res +=    -1.000000 * einsum('Iji,aj->Iai'       , cc_obj.h1p1e, amp[0])
    res +=     1.000000 * einsum('Iab,bi->Iai'       , cc_obj.h1p1e, amp[0])
    res +=    -1.000000 * einsum('Ijb,abji->Iai'     , cc_obj.h1p1e, amp[1])
    res +=    -1.000000 * einsum('jaib,Ibj->Iai'     , cc_obj.h2e, amp[3])
    res +=    -1.000000 * einsum('jb,aj,Ibi->Iai'    , cc_obj.h1e, amp[0], amp[3])
    res +=    -1.000000 * einsum('jb,bi,Iaj->Iai'    , cc_obj.h1e, amp[0], amp[3])
    res +=    -1.000000 * einsum('Ijb,bi,aj->Iai'    , cc_obj.h1p1e, amp[0], amp[0])
    res +=    -1.000000 * einsum('Jji,J,Iaj->Iai'    , cc_obj.h1p1e, amp[2], amp[3])
    res +=     1.000000 * einsum('Jab,J,Ibi->Iai'    , cc_obj.h1p1e, amp[2], amp[3])
    res +=    -1.000000 * einsum('Jjb,Iaj,Jbi->Iai'  , cc_obj.h1p1e, amp[3], amp[3])
    res +=    -1.000000 * einsum('Jjb,Ibi,Jaj->Iai'  , cc_obj.h1p1e, amp[3], amp[3])
    res +=     1.000000 * einsum('Jjb,Ibj,Jai->Iai'  , cc_obj.h1p1e, amp[3], amp[3])
    res +=    -1.000000 * einsum('jkib,aj,Ibk->Iai'  , cc_obj.h2e, amp[0], amp[3])
    res +=     1.000000 * einsum('jkib,bj,Iak->Iai'  , cc_obj.h2e, amp[0], amp[3])
    res +=     1.000000 * einsum('jabc,ci,Ibj->Iai'  , cc_obj.h2e, amp[0], amp[3])
    res +=    -1.000000 * einsum('jabc,cj,Ibi->Iai'  , cc_obj.h2e, amp[0], amp[3])
    res +=     1.000000 * einsum('jkbc,acji,Ibk->Iai', cc_obj.h2e, amp[1], amp[3])
    res +=     0.500000 * einsum('jkbc,ackj,Ibi->Iai', cc_obj.h2e, amp[1], amp[3])
    res +=     0.500000 * einsum('jkbc,cbji,Iak->Iai', cc_obj.h2e, amp[1], amp[3])
    res +=    -0.500000 * einsum('Jjb,aj,J,Ibi->Iai' , cc_obj.h1p1e, amp[0], amp[2], amp[3])
    res +=    -0.500000 * einsum('Jjb,bi,J,Iaj->Iai' , cc_obj.h1p1e, amp[0], amp[2], amp[3])
    res +=    -0.500000 * einsum('jkbc,aj,ck,Ibi->Iai', cc_obj.h2e, amp[0], amp[0], amp[3])
    res +=     0.500000 * einsum('jkbc,ci,aj,Ibk->Iai', cc_obj.h2e, amp[0], amp[0], amp[3])
    res +=    -0.500000 * einsum('jkbc,ci,bj,Iak->Iai', cc_obj.h2e, amp[0], amp[0], amp[3])
    return res

def ene(cc_obj, amp):
    """
    Generated by gen-cceqs.py at 2023-09-19 15:46:30
    Machine Info: Darwin 23.0.0
    Hostname:     Junjies-MacBook-Air.local
    Git Branch:   pauling
    Git Commit:   abc0d69e1b9e4f80a5539b87a023d6b3e073385f

    amp[0]: a+_v a_o
    amp[1]: a+_v a+_v a_o a_o
    amp[2]: b+
    amp[3]: b+ a+_v a_o

    ih = 0, ibra = 4, len(tmp.terms) = 0
    ih = 1, ibra = 4, len(tmp.terms) = 4
    ih = 2, ibra = 4, len(tmp.terms) = 2
    ih = 3, ibra = 4, len(tmp.terms) = 0
    """
    res  =     1.000000 * einsum('I,I->'             , cc_obj.h1p_eff, amp[2])
    res +=     1.000000 * einsum('ia,ai->'           , cc_obj.h1e, amp[0])
    res +=     1.000000 * einsum('Iia,Iai->'         , cc_obj.h1p1e, amp[3])
    res +=     0.250000 * einsum('ijab,baji->'       , cc_obj.h2e, amp[1])
    res +=     1.000000 * einsum('Iia,ai,I->'        , cc_obj.h1p1e, amp[0], amp[2])
    res +=    -0.500000 * einsum('ijab,bi,aj->'      , cc_obj.h2e, amp[0], amp[0])
    return res
