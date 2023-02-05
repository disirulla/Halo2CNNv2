use halo2_proofs::arithmetic::FieldExt;
use rand::{self, Rng};
use std::marker::PhantomData;
use std::ops::{AddAssign};
use std::vec;
use halo2_proofs::circuit::{Layouter, SimpleFloorPlanner, Value};
use halo2_proofs::plonk::{Advice, Circuit, Column, ConstraintSystem, Error, Instance, Selector, Expression, Constraints, TableColumn};
use halo2_proofs::poly::Rotation;
use halo2_proofs::{dev::MockProver, pasta::Fp};
use std::time::{Duration, Instant};

#[derive(Debug,Clone)]
pub struct TwoDVec<F: FieldExt> {
    data: Vec<Vec<Value<F>>>,
}

impl<F: FieldExt> TwoDVec<F> {
    pub fn new(a: Vec<Vec<Value<F>>>)->Self{

        Self{
            data: a,
     }
    }

}

// Instance Vector

#[derive(Debug,Clone)]
pub struct InstVector<F: FieldExt>{
    data: Vec<Column<Instance>>,
    len: usize,
    _marker: PhantomData<F>,

}

impl<F: FieldExt>  InstVector<F>{
    pub fn new_ins_vec(meta: &mut ConstraintSystem<F>, vec_size: usize, len: usize) -> InstVector<F>{

        let mut instbufarr = vec![];
        for _i in 0..vec_size{
                let buf = meta.instance_column();
                meta.enable_equality(buf);
                instbufarr.push(buf);
            }

            Self {
                data: instbufarr,
                len,
                _marker: PhantomData,
            }
        }
    }

// Advice Vector

#[derive(Debug,Clone)]
pub struct AdviceVector<F: FieldExt>{
   pub data: Vec<Column<Advice>>,
    len: usize,
    _marker: PhantomData<F>,
}

impl<F: FieldExt>  AdviceVector<F>{
    pub fn new_adv_vec(meta: &mut ConstraintSystem<F>, vec_size: usize, len: usize) -> AdviceVector<F>{
        let mut advbufarr = vec![];
        for _i in 0..vec_size{
                let buf = meta.advice_column();
                meta.enable_equality(buf);
                advbufarr.push(buf);
            }

            Self {
                data: advbufarr,
                len,
                _marker: PhantomData,

            }
        }
    }


    
#[derive(Debug,Clone)]
pub struct Layer <F: FieldExt>{
    pub image: AdviceVector<F>,
    imlen: usize,
    imwid: usize,
    pub kernel: AdviceVector<F>,
    kerlen: usize,
    kerwid: usize,
    pub inter: AdviceVector<F>,
    conlen: usize,
    conwid: usize,
    pub relu: AdviceVector<F>,
  pub  maxconvvalue: usize,
    _marker: PhantomData<F>,
}

impl<F: FieldExt>  Layer<F>{
    pub fn new_layer(meta: &mut ConstraintSystem<F>,imlen: usize, imwid: usize, kerlen: usize,
        kerwid: usize) -> Layer<F>
    {
        let image = AdviceVector::new_adv_vec(meta, imwid, imlen);
        let kernel = AdviceVector::new_adv_vec(meta, kerwid, kerlen);
        let conlen = imlen - kerlen + 1;
        let conwid = imwid - kerwid + 1;
        let inter = AdviceVector::new_adv_vec(meta, conwid, conlen);
        let relu = AdviceVector::new_adv_vec(meta, conwid, conlen);
        let maxconvvalue = 255*5*kerlen*kerwid;
        Self {
            image,
            imlen,
            imwid,
            kernel,
            kerlen,
            kerwid,
            inter,
            conlen,
            conwid,
            relu,
            maxconvvalue,
            _marker: PhantomData,
        }
    }
}

// Lookup Table: [0:MAXCONVVALUE]

// Contains all possible positive values, output of a ReLU function should belong to this table

#[derive(Debug, Clone)]
pub struct ReLULoookUp<F: FieldExt> {
      pub relop: TableColumn,
      r: usize,
        _marker: PhantomData<F>,
}
impl<F: FieldExt> ReLULoookUp<F> {
      pub  fn configure(meta: &mut ConstraintSystem<F>, r:usize) -> Self {
            let relop = meta.lookup_table_column();
    
            Self {
                relop,
                r,
                _marker: PhantomData,
            }
        }
      pub  fn load(&self, layouter: &mut impl Layouter<F>) -> Result<(), Error> {
            layouter.assign_table(
                || "relu lookup table",
                |mut table| {
                    let mut offset = 0;
                    for i in 0..=self.r {
                        let k = i;    
                        table.assign_cell(
                                || "relop",
                                self.relop ,
                                offset,
                                || Value::known(F::from(k as u64)),
                            )?;
                        
                    offset += 1; 
                    }
    
                    Ok(())
                },
            )
        }
    }

