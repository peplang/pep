use crate::ast::{Expressao, Instrucao, OpBinario, OpUnario, Programa};
use crate::vm::VmValor;
use std::collections::HashMap;

pub type Registro = u16;

#[derive(Debug, Clone)]
pub enum InstrucaoReg {
    Constante {
        destino: Registro,
        valor: VmValor,
    },
    Mover {
        destino: Registro,
        origem: Registro,
    },
    Binaria {
        destino: Registro,
        a: Registro,
        op: OpBinario,
        b: Registro,
    },
    Unaria {
        destino: Registro,
        op: OpUnario,
        origem: Registro,
    },
    IntervaloValido {
        destino: Registro,
        atual: Registro,
        fim: Registro,
        passo: Registro,
    },
    Imprimir(Vec<Registro>),
    Pular(usize),
    PularSeFalso {
        condicao: Registro,
        alvo: usize,
    },
    Parar,
}

#[derive(Debug, Clone)]
pub struct ProgramaReg {
    pub instrucoes: Vec<InstrucaoReg>,
    pub quantidade_registradores: usize,
}

pub fn compilar(programa: &Programa) -> Result<ProgramaReg, String> {
    let mut c = CompiladorReg {
        codigo: Vec::new(),
        variaveis: HashMap::new(),
        proximo: 0,
    };
    c.bloco(programa)?;
    c.codigo.push(InstrucaoReg::Parar);
    Ok(ProgramaReg {
        instrucoes: c.codigo,
        quantidade_registradores: c.proximo as usize,
    })
}

struct CompiladorReg {
    codigo: Vec<InstrucaoReg>,
    variaveis: HashMap<String, Registro>,
    proximo: Registro,
}

impl CompiladorReg {
    fn novo_registro(&mut self) -> Result<Registro, String> {
        let r = self.proximo;
        self.proximo = self
            .proximo
            .checked_add(1)
            .ok_or("limite de registradores excedido")?;
        Ok(r)
    }

    fn variavel(&mut self, nome: &str) -> Result<Registro, String> {
        if let Some(r) = self.variaveis.get(nome) {
            return Ok(*r);
        }
        let r = self.novo_registro()?;
        self.variaveis.insert(nome.to_string(), r);
        Ok(r)
    }

    fn bloco(&mut self, bloco: &[Instrucao]) -> Result<(), String> {
        for i in bloco {
            self.instrucao(i)?;
        }
        Ok(())
    }

    fn instrucao(&mut self, i: &Instrucao) -> Result<(), String> {
        match i {
            Instrucao::Localizada { instrucao, .. } => self.instrucao(instrucao),
            Instrucao::DeclararVar { nome, valor } => {
                let destino = self.variavel(nome)?;
                let origem = match valor {
                    Some(v) => self.expressao(v)?,
                    None => self.constante(VmValor::Null)?,
                };
                self.codigo.push(InstrucaoReg::Mover { destino, origem });
                Ok(())
            }
            Instrucao::Expressao(e) => {
                self.expressao(e)?;
                Ok(())
            }
            Instrucao::Imprimir(args) => {
                let regs = args
                    .iter()
                    .map(|e| self.expressao(e))
                    .collect::<Result<Vec<_>, _>>()?;
                self.codigo.push(InstrucaoReg::Imprimir(regs));
                Ok(())
            }
            Instrucao::Se {
                condicao,
                entao,
                senao,
            } => {
                let cond = self.expressao(condicao)?;
                let jf = self.codigo.len();
                self.codigo.push(InstrucaoReg::PularSeFalso {
                    condicao: cond,
                    alvo: 0,
                });
                self.bloco(entao)?;
                if let Some(senao) = senao {
                    let j = self.codigo.len();
                    self.codigo.push(InstrucaoReg::Pular(0));
                    let inicio_senao = self.codigo.len();
                    self.patch(jf, inicio_senao);
                    self.bloco(senao)?;
                    let fim = self.codigo.len();
                    self.patch(j, fim);
                } else {
                    let fim = self.codigo.len();
                    self.patch(jf, fim);
                }
                Ok(())
            }
            Instrucao::Enquanto { condicao, corpo } => {
                let inicio = self.codigo.len();
                let cond = self.expressao(condicao)?;
                let jf = self.codigo.len();
                self.codigo.push(InstrucaoReg::PularSeFalso {
                    condicao: cond,
                    alvo: 0,
                });
                self.bloco(corpo)?;
                self.codigo.push(InstrucaoReg::Pular(inicio));
                let fim = self.codigo.len();
                self.patch(jf, fim);
                Ok(())
            }
            Instrucao::ParaIntervalo {
                variavel,
                inicio,
                fim,
                passo,
                corpo,
            } => {
                let atual = self.variavel(variavel)?;
                let inicio_r = self.expressao(inicio)?;
                let fim_r = self.expressao(fim)?;
                let passo_r = match passo {
                    Some(p) => self.expressao(p)?,
                    None => self.constante(VmValor::Int(1))?,
                };
                self.codigo.push(InstrucaoReg::Mover {
                    destino: atual,
                    origem: inicio_r,
                });
                let topo = self.codigo.len();
                let valido = self.novo_registro()?;
                self.codigo.push(InstrucaoReg::IntervaloValido {
                    destino: valido,
                    atual,
                    fim: fim_r,
                    passo: passo_r,
                });
                let jf = self.codigo.len();
                self.codigo.push(InstrucaoReg::PularSeFalso {
                    condicao: valido,
                    alvo: 0,
                });
                self.bloco(corpo)?;
                self.codigo.push(InstrucaoReg::Binaria {
                    destino: atual,
                    a: atual,
                    op: OpBinario::Soma,
                    b: passo_r,
                });
                self.codigo.push(InstrucaoReg::Pular(topo));
                let depois = self.codigo.len();
                self.patch(jf, depois);
                Ok(())
            }
            _ => Err(
                "instrucao ainda nao suportada pela VM de registradores experimental".to_string(),
            ),
        }
    }

    fn expressao(&mut self, e: &Expressao) -> Result<Registro, String> {
        match e {
            Expressao::Inteiro(n) => self.constante(VmValor::Int(*n)),
            Expressao::Numero(n) => self.constante(VmValor::Num(*n)),
            Expressao::Texto(s) => self.constante(VmValor::Str(s.clone())),
            Expressao::Booleano(b) => self.constante(VmValor::Bool(*b)),
            Expressao::Nulo => self.constante(VmValor::Null),
            Expressao::Variavel(nome) => self
                .variaveis
                .get(nome)
                .copied()
                .ok_or_else(|| format!("variavel '{}' usada antes da declaracao", nome)),
            Expressao::Atribuicao { nome, valor } => {
                let origem = self.expressao(valor)?;
                let destino = self.variavel(nome)?;
                self.codigo.push(InstrucaoReg::Mover { destino, origem });
                Ok(destino)
            }
            Expressao::BinOp { esq, op, dir } => {
                let a = self.expressao(esq)?;
                let b = self.expressao(dir)?;
                let destino = self.novo_registro()?;
                self.codigo.push(InstrucaoReg::Binaria {
                    destino,
                    a,
                    op: op.clone(),
                    b,
                });
                Ok(destino)
            }
            Expressao::UnOp { op, expr } => {
                let origem = self.expressao(expr)?;
                let destino = self.novo_registro()?;
                self.codigo.push(InstrucaoReg::Unaria {
                    destino,
                    op: op.clone(),
                    origem,
                });
                Ok(destino)
            }
            _ => Err(
                "expressao ainda nao suportada pela VM de registradores experimental".to_string(),
            ),
        }
    }

    fn constante(&mut self, valor: VmValor) -> Result<Registro, String> {
        let destino = self.novo_registro()?;
        self.codigo.push(InstrucaoReg::Constante { destino, valor });
        Ok(destino)
    }

    fn patch(&mut self, pos: usize, alvo: usize) {
        match &mut self.codigo[pos] {
            InstrucaoReg::Pular(a) | InstrucaoReg::PularSeFalso { alvo: a, .. } => *a = alvo,
            _ => {}
        }
    }
}

pub struct MaquinaReg {
    registradores: Vec<VmValor>,
    ip: usize,
}

impl MaquinaReg {
    pub fn executar(programa: &ProgramaReg) -> Result<Self, String> {
        let mut vm = Self {
            registradores: vec![VmValor::Null; programa.quantidade_registradores],
            ip: 0,
        };
        let limite = std::env::var("PEP_MAX_OPS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(10_000_000);
        let mut operacoes = 0u64;
        while let Some(op) = programa.instrucoes.get(vm.ip).cloned() {
            operacoes = operacoes.saturating_add(1);
            if limite > 0 && operacoes > limite {
                return Err(format!(
                    "limite de {} operacoes excedido na VM de registradores",
                    limite
                ));
            }
            vm.ip += 1;
            match op {
                InstrucaoReg::Constante { destino, valor } => vm.set(destino, valor),
                InstrucaoReg::Mover { destino, origem } => {
                    let v = vm.get(origem)?.clone();
                    vm.set(destino, v);
                }
                InstrucaoReg::Binaria { destino, a, op, b } => {
                    let v = binaria(vm.get(a)?.clone(), op, vm.get(b)?.clone())?;
                    vm.set(destino, v);
                }
                InstrucaoReg::Unaria {
                    destino,
                    op,
                    origem,
                } => {
                    let v = match (op, vm.get(origem)?.clone()) {
                        (OpUnario::Negativo, VmValor::Int(n)) => VmValor::Int(-n),
                        (OpUnario::Negativo, VmValor::Num(n)) => VmValor::Num(-n),
                        (OpUnario::Nao, v) => VmValor::Bool(!verdadeiro(&v)),
                        _ => return Err("operacao unaria invalida".to_string()),
                    };
                    vm.set(destino, v);
                }
                InstrucaoReg::IntervaloValido {
                    destino,
                    atual,
                    fim,
                    passo,
                } => {
                    let a = numero(vm.get(atual)?)?;
                    let f = numero(vm.get(fim)?)?;
                    let p = numero(vm.get(passo)?)?;
                    vm.set(
                        destino,
                        VmValor::Bool(if p >= 0.0 { a <= f } else { a >= f }),
                    );
                }
                InstrucaoReg::Imprimir(regs) => {
                    println!(
                        "{}",
                        regs.iter()
                            .map(|r| vm.get(*r).map(ToString::to_string))
                            .collect::<Result<Vec<_>, _>>()?
                            .join(" ")
                    );
                }
                InstrucaoReg::Pular(alvo) => vm.ip = alvo,
                InstrucaoReg::PularSeFalso { condicao, alvo } => {
                    if !verdadeiro(vm.get(condicao)?) {
                        vm.ip = alvo;
                    }
                }
                InstrucaoReg::Parar => break,
            }
        }
        Ok(vm)
    }

    fn get(&self, r: Registro) -> Result<&VmValor, String> {
        self.registradores
            .get(r as usize)
            .ok_or_else(|| format!("registro r{} invalido", r))
    }
    fn set(&mut self, r: Registro, v: VmValor) {
        self.registradores[r as usize] = v;
    }
}

fn numero(v: &VmValor) -> Result<f64, String> {
    match v {
        VmValor::Int(n) => Ok(*n as f64),
        VmValor::Num(n) => Ok(*n),
        _ => Err("numero esperado".to_string()),
    }
}

fn verdadeiro(v: &VmValor) -> bool {
    match v {
        VmValor::Null => false,
        VmValor::Bool(b) => *b,
        VmValor::Int(n) => *n != 0,
        VmValor::Num(n) => *n != 0.0,
        VmValor::Str(s) => !s.is_empty(),
        _ => true,
    }
}

fn binaria(a: VmValor, op: OpBinario, b: VmValor) -> Result<VmValor, String> {
    use OpBinario::*;
    match op {
        Soma => match (a, b) {
            (VmValor::Int(x), VmValor::Int(y)) => Ok(VmValor::Int(x + y)),
            (VmValor::Str(x), y) => Ok(VmValor::Str(x + &y.to_string())),
            (x, VmValor::Str(y)) => Ok(VmValor::Str(x.to_string() + &y)),
            (x, y) => Ok(VmValor::Num(numero(&x)? + numero(&y)?)),
        },
        Subtracao => Ok(VmValor::Num(numero(&a)? - numero(&b)?)),
        Multiplicacao => Ok(VmValor::Num(numero(&a)? * numero(&b)?)),
        Divisao => {
            let d = numero(&b)?;
            if d == 0.0 {
                Err("divisao por zero".to_string())
            } else {
                Ok(VmValor::Num(numero(&a)? / d))
            }
        }
        DivisaoInteira => {
            let d = numero(&b)?;
            if d == 0.0 {
                Err("divisao por zero".to_string())
            } else {
                Ok(VmValor::Int((numero(&a)? / d).floor() as i64))
            }
        }
        Modulo => Ok(VmValor::Num(numero(&a)? % numero(&b)?)),
        Igual => Ok(VmValor::Bool(a == b)),
        DiferenteDe => Ok(VmValor::Bool(a != b)),
        MenorQue => Ok(VmValor::Bool(numero(&a)? < numero(&b)?)),
        MaiorQue => Ok(VmValor::Bool(numero(&a)? > numero(&b)?)),
        MenorOuIgual => Ok(VmValor::Bool(numero(&a)? <= numero(&b)?)),
        MaiorOuIgual => Ok(VmValor::Bool(numero(&a)? >= numero(&b)?)),
        E => Ok(VmValor::Bool(verdadeiro(&a) && verdadeiro(&b))),
        Ou => Ok(VmValor::Bool(verdadeiro(&a) || verdadeiro(&b))),
        _ => Err("operador ainda nao suportado pela VM de registradores".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{lexer::Lexer, parser::Parser};
    #[test]
    fn executa_aritmetica_e_laco() {
        let fonte = "var soma = 0\npara i de 1 ate 5 { soma = soma + i }";
        let ast = Parser::novo(Lexer::novo(fonte).tokenizar().unwrap())
            .parsear()
            .unwrap();
        let programa = compilar(&ast).unwrap();
        let vm = MaquinaReg::executar(&programa).unwrap();
        let reg = programa.instrucoes.iter().find_map(|_| None::<u16>);
        assert!(reg.is_none());
        assert!(vm
            .registradores
            .iter()
            .any(|v| *v == VmValor::Int(15) || *v == VmValor::Num(15.0)));
    }
}
