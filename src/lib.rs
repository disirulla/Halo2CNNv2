use halo2_proofs::arithmetic::FieldExt;
use rand::{self, Rng};
use std::marker::PhantomData;
use std::ops::{AddAssign, Mul, Add};
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


pub struct InputGenerator<Fp>{
    pub image: Vec<Vec<Value<Fp>>>,
    pub kernels: Vec<Vec<Vec<Value<Fp>>>>,
    pub op: Vec<Vec<Value<Fp>>>,
}

impl  InputGenerator<Fp>{
    pub fn new_input(DIMS:Vec<[usize;4]>, MAX_CONV_VALUE:usize) -> Self
    {
        let mut rng = rand::thread_rng();
        
        // Kernels
        let mut kernels = vec![];
        for i in 0..DIMS.len(){
            kernels.push(vec![]);
          for j in 0..DIMS[i][3]{
                kernels[i].push(vec![]);
                for k in 0..DIMS[i][2]{
                    let mut buf = Value::known(Fp::zero());
                    let random_value:f32 = rng.gen_range(-5.0..=5.0);
                    let x = random_value.round() as i32;
                    if x < 0
                    { buf = Value::known(-Fp::from(x as u64));} 
                    else 
                    {buf = Value::known(Fp::from(x as u64));} 
                    kernels[i][j].push(buf);    
                }
                }
             }
        
        
    

        // Random Image
        let mut  init_image = Vec::new();
        for j in 0..DIMS[0][1]{
            init_image.push(vec![]);
            for k in 0..DIMS[0][0]{
                let x = rng.gen_range(0..=255);
                let mut buf = Value::known(Fp::from(x));
                init_image[j].push(buf);    
            }
        }


        
    let mut image = init_image.clone();
    for m in 0..DIMS.len(){

        let filter = kernels[m].clone();
        let conwid = DIMS[m][1] - DIMS[m][3] +1;
        let conlen = DIMS[m][0] - DIMS[m][2] +1;

        let mut convimage = vec![];
        let max_pos_val = Value::known(Fp::from(MAX_CONV_VALUE as u64));
        let zero = Fp::zero();
        for i in 0..conwid{
            convimage.push(Vec::new());
            for j in 0..conlen {
                let mut conval = Value::known(Fp::zero());
                for k in 0..DIMS[m][3]{
                    for l in 0..DIMS[m][2]{
                        conval = conval.add(image[i+k][j+l].clone().mul(filter[k][l].clone()));
                    }
                }
            conval.clone().zip(max_pos_val.clone()).map(|(a,b)|{
                let  zero = Fp::zero(); 
                        if a.gt(&b) 
                        {zero} 
                        else 
                        {a}
            }); //relu
            
            
            convimage[i].push(conval);
            }
        }
    image = convimage.clone()
    
    }

    let mut op = image.clone();

    Self{
        image: init_image,
        kernels,
        op
    }

    }
} 