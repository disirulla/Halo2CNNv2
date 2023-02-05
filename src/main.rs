mod lib;
use halo2_proofs::circuit::{Chip, AssignedCell};
use lib::{TwoDVec,Layer, ReLULoookUp};
use rand::Rng;
use std::marker::PhantomData;
use std::ops::{Neg, Add};
use halo2_proofs::arithmetic::FieldExt;
use halo2_proofs::circuit::{Layouter, SimpleFloorPlanner, Value};
use halo2_proofs::plonk::{Advice, Circuit, Column, ConstraintSystem, Error, Instance, Selector, Expression, Constraints, TableColumn};
use halo2_proofs::poly::Rotation;
use halo2_proofs::{dev::MockProver, pasta::Fp};
use std::time::{Duration, Instant};

static CONLAYERSCOUNT:usize = 2; 
static L1DIMS: [usize; 4] = [4,4,2,2]; 
static L2DIMS: [usize; 4] = [3,3,2,2];
static DIMS:[[usize;4];2] = [L1DIMS, L2DIMS];
static MAXCONVVALUE:usize = DIMS[0][2]*DIMS[0][3]*5*255;


#[derive(Debug,Clone)]
struct CNN<F: FieldExt>{
    conlayers: Vec<Layer<F>>,
    seldot: Vec<Selector>,
    selrel: Vec<Selector>,
    reltable: ReLULoookUp<F>,
    _marker: PhantomData<F>,
}


#[derive(Debug,Clone)]
struct CNNChip<F: FieldExt>{
    config: CNN<F>,
    _marker: PhantomData<F>,
}

impl<F: FieldExt> CNNChip<F>{
    
    pub fn configure(meta: &mut ConstraintSystem<F>) -> CNN<F> {
        
        let mut conlayers = vec![];
        let mut seldot = vec![];
        let mut selrel = vec![];
        let reltable = ReLULoookUp::configure(meta, MAXCONVVALUE);

        for l in 0..CONLAYERSCOUNT{
            println!("Layer {} with these following dims {}, {}",l, DIMS[l][0], DIMS[l][2]);
            let buflayer = Layer::new_layer(meta,DIMS[l][0], DIMS[l][1], DIMS[l][2], DIMS[l][3]);
            conlayers.push(buflayer);
        
            seldot.push(meta.selector());
            selrel.push(meta.complex_selector());
        
            let conwid = DIMS[l][1] - DIMS[l][3] +1;
            let conlen = DIMS[l][0] - DIMS[l][2] +1;
        meta.create_gate("conv", |meta|{

            let s = meta.query_selector(seldot[l]);
            let mut diff = vec![];
            
                
                let mut imgcells = vec![];
                for i in 0..DIMS[l][1]{
                    imgcells.push(Vec::new());
                    for j in 0..DIMS[l][0]{
                        let buf = meta.query_advice(conlayers[l].image.data[i], Rotation(j as i32));
                        imgcells[i].push(buf);
                    }
                }

                let mut kercells = vec![];
                for i in 0..DIMS[l][3]{
                    kercells.push(Vec::new());
                    for j in 0..DIMS[l][2]{
                        let buf = meta.query_advice(conlayers[l].kernel.data[i], Rotation(j as i32));
                        kercells[i].push(buf);
                    }
                }

                

                let mut concells = vec![];
                for i in 0..conwid{
                concells.push(Vec::new());
                for j in 0..conlen{
                    let buf = meta.query_advice(conlayers[l].inter.data[i], Rotation(j as i32));
                    concells[i].push(buf);
                     }
                }

                let mut condash = vec![];
                for i in 0..conwid{
                    condash.push(vec![]);
                    for j in 0..conlen{
                        let mut conval = Expression::Constant(F::zero());                 
                        // let mut conval = Expression::Constant(F::one());   // A bug                
                        for k in 0..DIMS[l][3]{
                            for l in 0..DIMS[l][2]{
                                conval = conval + (imgcells[i+k][j+l].clone()*kercells[k][l].clone());
                            }
                        }
                condash[i].push(conval);   
                let buf = condash[i][j].clone() - (concells[i][j].clone());
                diff.push(buf);
                }
            }
            
        Constraints::with_selector(s, diff)   
        });
    
        for i in 0..conwid{
            for j in 0..conlen{
                meta.lookup(|meta| {
                    let selrel = meta.query_selector(selrel[l]);
                    let valueop = meta.query_advice(conlayers[l].relu.data[i], Rotation(j as i32));
                    vec![(selrel*valueop , reltable.relop)]
              
                 });
                }}

            }
        CNN {
            conlayers,
            seldot,
            selrel,
            reltable,
            _marker: PhantomData, }
    }

}





#[derive(Debug,Clone)]

struct CNNCircuit<F: FieldExt>{
    initial_image: Vec<Vec<Value<F>>>,
    kernels: Vec<Vec<Vec<Value<F>>>>,
}

impl<F: FieldExt> Circuit<F> for CNNCircuit<F>{
    
    type Config = CNN<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        CNNChip::configure(meta)
    }

    fn synthesize(&self, config: Self::Config, mut layouter: impl Layouter<F>) -> Result<(), Error> {
        config.reltable.load(&mut layouter)?;
        let mut img = self.initial_image.clone();
        let mut bufimg = vec![];

    for i in 0..CONLAYERSCOUNT{

        let ker = self.kernels[i].clone();
        let cnn = layouter.assign_region(|| "layer".to_owned()+&i.to_string(), |mut region|{
            

            config.seldot[i].enable(&mut region, 0)?;
            config.selrel[i].enable(&mut region, 0)?;
            
            // Image Load
            let mut imgcells = vec![];
            for j in 0..DIMS[i][1]{
                imgcells.push(Vec::new());
                for k in 0..DIMS[i][0]{

                let i_cell = region.assign_advice(||"image".to_owned()+&j.to_string()+&k.to_string(),
                 config.conlayers[i].image.data[j], 
                 k, 
                 || img[j][k])?;
                imgcells[j].push(i_cell);   
                };
            };


            // Kernel Load
            let mut kercells = vec![];
            for j in 0..DIMS[i][3]{
                kercells.push(Vec::new());
                for k in 0..DIMS[i][2]{
                let k_cell = region.assign_advice(||"kernel".to_owned()+&j.to_string()+&k.to_string(),
                 config.conlayers[i].kernel.data[j], 
                 k, 
                 || ker[j][k])?;
                kercells[j].push(k_cell);   
                };
            };

            let conwid = DIMS[i][1] - DIMS[i][3] +1;
            let conlen = DIMS[i][0] - DIMS[i][2] +1;

            // Convolution 
            let mut convcells = vec![];
            for m in 0..conwid{
                convcells.push(vec![]);
                for j in 0..conlen {
                    let mut conval = Value::known(F::zero());                    
                    for k in 0..DIMS[i][3]{
                        for l in 0..DIMS[i][2]{
                            conval = conval.add(imgcells[m+k][j+l].value().copied()*kercells[k][l].value());
                        };
                    };

                
                   
                let con_cell = region.assign_advice(||"conv".to_owned()+&m.to_string()+&j.to_string(),
                 config.conlayers[i].inter.data[m], 
                 j, 
                 || conval)?;
                convcells[m].push(con_cell);   
            };
            
        };


            // ReLU
            let mut relcells = vec![];
            for m in 0..conwid{
                relcells.push(vec![]);
                bufimg.push(vec![]);
                for j in 0..conlen {
                    let maxconvval = Value::known(F::from(MAXCONVVALUE as u64));                 
                    let rel_val = convcells[m][j].clone().value().copied().zip(maxconvval).map(|(a,b)| {
                        let  zero = F::zero(); 
                        if a.gt(&b) 
                        {zero} 
                        else 
                        {a}
                    } );
                bufimg[m].push(rel_val.clone());

                let rel_cell = region.assign_advice(||"conv".to_owned()+&m.to_string()+&j.to_string(),
                 config.conlayers[i].relu.data[m], 
                 j, 
                 || rel_val)?;
                relcells[m].push(rel_cell);

            };
            
        };

            Ok(relcells)


        });
        img = bufimg.clone();
        }
        
        Ok(())
    }

    fn without_witnesses(&self) -> Self {
        return self.clone();
    }
}

fn main() {
    let k = 18; // Alter based on # of rows

        let mut rng = rand::thread_rng();
        
        // Kernels
        let mut kernels = vec![];
        for i in 0..CONLAYERSCOUNT{
            kernels.push(vec![]);
          for j in 0..DIMS[i][3]{
                kernels[i].push(vec![]);
                for k in 0..DIMS[i][2]{
                    let mut buf = Value::known(Fp::zero());
                    let random_value:f32 = rng.gen_range(0.0..=5.0);
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
        let mut image = Vec::new();
        for j in 0..DIMS[0][1]{
            image.push(vec![]);
            for k in 0..DIMS[0][0]{
                let x = rng.gen_range(0..=255);
                let mut buf = Value::known(Fp::from(x));
                image[j].push(buf);    
            }
        }


        let circuit = CNNCircuit {
            initial_image: image,
            kernels,
        };


        // MockProver
        let start = Instant::now();
        let prover = MockProver::run(k, &circuit, vec![]);
        let duration = start.elapsed();

        // prover.unwrap().assert_satisfied();
        match prover.unwrap().verify(){
            Ok(()) => { println!("Yes proved!")},
            Err(_) => {println!("Not proved!")}

        }
        println!("Time taken by MockProver: {:?}", duration);

            
}
