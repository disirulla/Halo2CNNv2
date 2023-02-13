mod lib;
use halo2_proofs::circuit::{Chip, AssignedCell};
use lib::{TwoDVec,Layer, ReLULoookUp};
use rand::Rng;
use std::marker::PhantomData;
use std::ops::{Neg, Add};
use halo2_proofs::arithmetic::FieldExt;
use halo2_proofs::arithmetic::*;
use halo2_proofs::circuit::floor_planner::V1;
use halo2_proofs::circuit::{Layouter, Value}; 
use halo2_proofs::plonk::{Advice, Circuit, Column, ConstraintSystem, Error, Instance, Selector, Expression, Constraints, TableColumn};
use halo2_proofs::poly::Rotation;
use halo2_proofs::{dev::MockProver, pasta::Fp};
use std::time::{Duration, Instant};


static CONLAYERSCOUNT:usize = 3; 
static L1DIMS: [usize; 4] = [28,28,13,13]; 
static L2DIMS: [usize; 4] = [16,16,13,13];
static L3DIMS: [usize; 4] = [4,4,2,2];
// static L4DIMS: [usize; 4] = [4,4,2,2];
// static L5DIMS: [usize; 4] = [3,3,1,1];
static DIMS:[[usize;4];3] = [L1DIMS, L2DIMS, L3DIMS];
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
    
    // pub fn dotgate(l:usize, off:usize,imw:usize, iml:usize, kerw:usize, kerl:usize,meta: &mut ConstraintSystem<F>, conlayers:Vec<Layer<F>>){
    //     meta.create_gate("dot", |meta|{
    //         let im = vec![];
    //         for i in 0..imw{
    //             for j in 0..iml{
    //                 let buf = meta.query_advice(conlayers[l].image.data[i], Rotation(j as i32));
    //             }
    //         }
    //     });
    // }

    pub fn configure(meta: &mut ConstraintSystem<F>) -> CNN<F> {
        
        let mut conlayers = vec![];
        let mut seldot = vec![];
        let mut selrel = vec![];
        let reltable = ReLULoookUp::configure(meta, MAXCONVVALUE);

        for l in 0..CONLAYERSCOUNT{
            let buflayer = Layer::new_layer(meta,DIMS[l][0], DIMS[l][1], DIMS[l][2], DIMS[l][3]);
            conlayers.push(buflayer);
        
            
            // seldot.push(vec![]);
            seldot.push(meta.selector());
            selrel.push(meta.complex_selector());
        
            let conwid = DIMS[l][1] - DIMS[l][3] +1;
            let conlen = DIMS[l][0] - DIMS[l][2] +1;
            // for i in 0..conwid{
            //     seldot[l].push(meta.selector());
            // }

            for k in 0..conwid{
                for n in 0..conlen{
                    meta.create_gate("conv", |meta|{
                        let s = meta.query_selector(seldot[l]);
                        let mut diff = vec![];
        
                        let mut imgcells = vec![];
                        let mut kercells = vec![];
                        for i in 0..DIMS[l][3]{
                            imgcells.push(vec![]);
                            kercells.push(vec![]);

                            for j in 0..DIMS[l][2]{
                               let bufimg = meta.query_advice(conlayers[l].image.data[i+k], Rotation((n+j) as i32));
                                imgcells[i].push(bufimg.clone());

                                let bufker = meta.query_advice(conlayers[l].kernel.data[i], Rotation(j as i32));
                                kercells[i].push(bufker.clone());
                            }
                        }
                        
                        // let kercells = vec![];
                        // for 
    
                        let concell = meta.query_advice(conlayers[l].inter.data[k], Rotation(n as i32));
        
                        let mut conval = Expression::Constant(F::zero());                 
                        // let mut conval = Expression::Constant(F::one());   // A bug                
                        for o in 0..DIMS[l][3]{
                            for p in 0..DIMS[l][2]{
                                // let fil_val = meta.query_advice(conlayers[l].kernel.data[o], Rotation(p as i32));
                                conval = conval + (imgcells[o][p].clone()*kercells[o][p].clone());
                            }
                        }
                        diff.push(conval.clone()-concell.clone());
        
                        Constraints::with_selector(s, diff)  }
                    );
                }
            }

            
    
        // for i in 0..conwid{
        //     for j in 0..conlen{
        //         meta.lookup(|meta| {
        //             let selrel = meta.query_selector(selrel[l]);
        //             let valueop = meta.query_advice(conlayers[l].relu.data[i], Rotation(j as i32));
        //             vec![(selrel*valueop , reltable.relop)]
              
        //          });
        //         }}

            
        }
        
         return CNN {
            conlayers,
            seldot,
            selrel,
            reltable,
            _marker: PhantomData, };
    
    }

}



#[derive(Debug,Clone)]

struct CNNCircuit<F: FieldExt>{
    initial_image: Vec<Vec<Value<F>>>,
    kernels: Vec<Vec<Vec<Value<F>>>>,
}

impl<F: FieldExt> Circuit<F> for CNNCircuit<F>{
    
    type Config = CNN<F>;
    type FloorPlanner = V1;

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
            config.seldot[i].enable(&mut region, 0)?;

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
                    config.selrel[i].enable(&mut region, j)?;
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

            Ok(imgcells)


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
    
        use lib::InputGenerator;
        let gen = InputGenerator::new_input(DIMS.to_vec(), MAXCONVVALUE);

        let init_image = gen.image;
        let kernels = gen.kernels;


        let circuit = CNNCircuit {
            initial_image: init_image,
            kernels,
        };

        // MockProver
        let start = Instant::now();
        let prover = MockProver::run(k, &circuit, vec![]);
        let duration = start.elapsed();

        
        prover.unwrap().assert_satisfied();
        // match prover.unwrap().verify(){
        //     Ok(()) => { println!("Yes proved!")},
        //     Err(_) => {println!("Not proved!")}

        // }
        println!("Time taken by MockProver: {:?}", duration);

       
            
}
