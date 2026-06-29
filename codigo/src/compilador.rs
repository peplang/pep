/// Compilador PEP: AST -> bytecode (Vec<Op>)
use crate::ast::*;
use crate::bytecode::Op;
use std::sync::Arc;

#[inline]
fn arc(s: &str) -> Arc<str> {
    Arc::from(s)
}

pub fn compilar(programa: &Programa) -> Result<Vec<Op>, String> {
    let mut ctx = Compilador::novo();
    ctx.compilar_bloco(programa, None, &mut Vec::new())?;
    ctx.emit(Op::Halt);
    Ok(ctx.ops)
}

/// Compila o corpo de uma função (termina com ReturnNull, não Halt).
pub fn compilar_corpo_funcao(corpo: &[crate::ast::Instrucao]) -> Result<Vec<Op>, String> {
    let mut ctx = Compilador::novo();
    ctx.compilar_bloco(corpo, None, &mut Vec::new())?;
    ctx.emit(Op::ReturnNull);
    Ok(ctx.ops)
}

struct Compilador {
    ops: Vec<Op>,
}

impl Compilador {
    fn novo() -> Self {
        Compilador { ops: Vec::new() }
    }

    fn emit(&mut self, op: Op) -> usize {
        self.ops.push(op);
        self.ops.len() - 1
    }

    fn len(&self) -> usize {
        self.ops.len()
    }

    fn patch(&mut self, idx: usize, alvo: usize) {
        match &mut self.ops[idx] {
            Op::Jump(t) | Op::JumpFalse(t) | Op::JumpTrue(t) | Op::TryCatch(t) => *t = alvo,
            _ => {}
        }
    }

    // -- Compilacao de bloco ---------------------------------------------------

    fn compilar_bloco(
        &mut self,
        instrucoes: &[Instrucao],
        loop_ini: Option<usize>,
        breaks: &mut Vec<usize>,
    ) -> Result<(), String> {
        for instr in instrucoes {
            self.compilar_instrucao(instr, loop_ini, breaks)?;
        }
        Ok(())
    }

    fn compilar_instrucao(
        &mut self,
        instr: &Instrucao,
        loop_ini: Option<usize>,
        breaks: &mut Vec<usize>,
    ) -> Result<(), String> {
        match instr {
            Instrucao::Localizada { instrucao, .. } => {
                self.compilar_instrucao(instrucao, loop_ini, breaks)?;
            }

            Instrucao::Expressao(expr) => {
                self.compilar_expr(expr)?;
                self.emit(Op::Pop);
            }

            Instrucao::DeclararVar { nome, valor } => {
                match valor {
                    Some(e) => self.compilar_expr(e)?,
                    None => {
                        self.emit(Op::PushNull);
                    }
                }
                self.emit(Op::Store(arc(nome)));
            }

            Instrucao::Imprimir(args) => {
                let n = args.len();
                for a in args {
                    self.compilar_expr(a)?;
                }
                self.emit(Op::Print(n));
            }

            Instrucao::Se {
                condicao,
                entao,
                senao,
            } => {
                self.compilar_expr(condicao)?;
                let jf = self.emit(Op::JumpFalse(0));
                self.compilar_bloco(entao, loop_ini, breaks)?;
                if let Some(bloco_senao) = senao {
                    let jmp = self.emit(Op::Jump(0));
                    self.patch(jf, self.len());
                    self.compilar_bloco(bloco_senao, loop_ini, breaks)?;
                    self.patch(jmp, self.len());
                } else {
                    self.patch(jf, self.len());
                }
            }

            Instrucao::Enquanto { condicao, corpo } => {
                let ini = self.len();
                self.compilar_expr(condicao)?;
                let jf = self.emit(Op::JumpFalse(0));
                let mut inner_breaks: Vec<usize> = Vec::new();
                self.compilar_bloco(corpo, Some(ini), &mut inner_breaks)?;
                self.emit(Op::Jump(ini));
                let fim = self.len();
                self.patch(jf, fim);
                for b in inner_breaks {
                    self.patch(b, fim);
                }
            }

            Instrucao::Para {
                variavel,
                iteravel,
                corpo,
            } => {
                let iter_var = arc(&format!("__iter_{}", self.len()));
                let idx_var = arc(&format!("__idx_{}", self.len()));

                self.compilar_expr(iteravel)?;
                self.emit(Op::Store(iter_var.clone()));
                self.emit(Op::PushNum(0.0));
                self.emit(Op::Store(idx_var.clone()));

                let ini = self.len();

                self.emit(Op::Load(iter_var.clone()));
                self.emit(Op::CallNative(arc("tamanho"), 1));
                self.emit(Op::Load(idx_var.clone()));
                self.emit(Op::Le);
                let jf = self.emit(Op::JumpTrue(0));

                self.emit(Op::Load(iter_var.clone()));
                self.emit(Op::Load(idx_var.clone()));
                self.emit(Op::GetIndex);
                self.emit(Op::Store(arc(variavel)));

                let mut inner_breaks: Vec<usize> = Vec::new();
                self.compilar_bloco(corpo, Some(ini), &mut inner_breaks)?;

                self.emit(Op::Load(idx_var.clone()));
                self.emit(Op::PushNum(1.0));
                self.emit(Op::Add);
                self.emit(Op::Store(idx_var.clone()));

                self.emit(Op::Jump(ini));
                let fim = self.len();
                self.patch(jf, fim);
                for b in inner_breaks {
                    self.patch(b, fim);
                }
            }

            Instrucao::ParaIntervalo {
                variavel,
                inicio,
                fim,
                passo,
                corpo,
            } => {
                self.compilar_expr(inicio)?;
                self.compilar_expr(fim)?;
                match passo {
                    Some(p) => self.compilar_expr(p)?,
                    None => {
                        self.emit(Op::PushNum(1.0));
                    }
                }
                self.emit(Op::IterStart { var: arc(variavel) });

                let jf_inicial = self.emit(Op::JumpFalse(0));
                let pula_stub = self.emit(Op::Jump(0));
                let continue_stub = self.emit(Op::Jump(0));

                let body_ini = self.len();
                self.patch(pula_stub, body_ini);

                let mut inner_breaks: Vec<usize> = Vec::new();
                self.compilar_bloco(corpo, Some(continue_stub), &mut inner_breaks)?;

                let iter_next_pos = self.len();
                self.emit(Op::IterNext {
                    var: arc(variavel),
                    loop_ini: body_ini,
                });

                self.patch(continue_stub, iter_next_pos);
                let fim_loop = self.len();
                self.patch(jf_inicial, fim_loop);
                for b in inner_breaks {
                    self.patch(b, fim_loop);
                }
            }

            Instrucao::Escolher {
                expr,
                casos,
                padrao,
            } => {
                let sw_var = arc(&format!("__sw_{}", self.len()));
                self.compilar_expr(expr)?;
                self.emit(Op::Store(sw_var.clone()));

                let mut jumps_ao_fim: Vec<usize> = Vec::new();

                for (valores_caso, bloco) in casos {
                    let mut vai_ao_bloco: Vec<usize> = Vec::new();
                    for val in valores_caso {
                        self.emit(Op::Load(sw_var.clone()));
                        self.compilar_expr(val)?;
                        self.emit(Op::Eq);
                        let jt = self.emit(Op::JumpTrue(0));
                        vai_ao_bloco.push(jt);
                    }
                    let pula_bloco = self.emit(Op::Jump(0));
                    let bloco_ini = self.len();
                    for jt in vai_ao_bloco {
                        self.patch(jt, bloco_ini);
                    }
                    self.compilar_bloco(bloco, loop_ini, breaks)?;
                    let jmp_fim = self.emit(Op::Jump(0));
                    jumps_ao_fim.push(jmp_fim);
                    let apos_bloco = self.len();
                    self.patch(pula_bloco, apos_bloco);
                }

                if let Some(bloco_padrao) = padrao {
                    self.compilar_bloco(bloco_padrao, loop_ini, breaks)?;
                }

                let fim = self.len();
                for j in jumps_ao_fim {
                    self.patch(j, fim);
                }
            }

            Instrucao::Funcao {
                nome,
                parametros,
                corpo,
            } => {
                let mut sub = Compilador::novo();
                sub.compilar_bloco(corpo, None, &mut Vec::new())?;
                sub.emit(Op::ReturnNull);
                self.emit(Op::DefFunc {
                    nome: arc(nome),
                    params: parametros.iter().map(|p| arc(&p.nome)).collect(),
                    corpo: sub.ops,
                });
                self.emit(Op::Store(arc(nome)));
            }

            Instrucao::Retornar(expr) => match expr {
                Some(e) => {
                    self.compilar_expr(e)?;
                    self.emit(Op::Return);
                }
                None => {
                    self.emit(Op::ReturnNull);
                }
            },

            Instrucao::Pare => {
                let b = self.emit(Op::Jump(0));
                breaks.push(b);
            }

            Instrucao::Continue => {
                if let Some(ini) = loop_ini {
                    self.emit(Op::Jump(ini));
                } else {
                    return Err("'continue' fora de um laco".to_string());
                }
            }

            Instrucao::Tentar {
                corpo,
                capturar,
                finalmente,
            } => {
                let try_op = self.emit(Op::TryCatch(0));
                self.compilar_bloco(corpo, loop_ini, breaks)?;
                self.emit(Op::EndTry);
                let pula_catch = self.emit(Op::Jump(0));

                let catch_ini = self.len();
                self.patch(try_op, catch_ini);
                if let Some((nome_var, bloco)) = capturar {
                    self.emit(Op::Store(arc(nome_var)));
                    self.compilar_bloco(bloco, loop_ini, breaks)?;
                } else {
                    self.emit(Op::Pop);
                }
                self.patch(pula_catch, self.len());

                if let Some(bloco) = finalmente {
                    self.compilar_bloco(bloco, loop_ini, breaks)?;
                }
            }

            Instrucao::Lancar(expr) => {
                self.compilar_expr(expr)?;
                self.emit(Op::Throw);
            }

            Instrucao::Importar { caminho, alias } => {
                self.emit(Op::Import {
                    caminho: caminho.clone(),
                    alias: alias.clone(),
                });
            }
            Instrucao::Incluir {
                caminho,
                obrigatorio,
            } => {
                self.emit(Op::Include {
                    caminho: caminho.clone(),
                    obrigatorio: *obrigatorio,
                });
            }
        }
        Ok(())
    }

    // -- Compilacao de expressao -----------------------------------------------

    fn compilar_expr(&mut self, expr: &Expressao) -> Result<(), String> {
        match expr {
            Expressao::Inteiro(n) => {
                self.emit(Op::PushInt(*n));
            }
            Expressao::Numero(n) => {
                self.emit(Op::PushNum(*n));
            }
            Expressao::Texto(s) => {
                self.emit(Op::PushStr(s.clone()));
            }
            Expressao::Booleano(b) => {
                self.emit(Op::PushBool(*b));
            }
            Expressao::Nulo => {
                self.emit(Op::PushNull);
            }

            Expressao::FuncaoAnonima { parametros, corpo } => {
                let mut sub = Compilador::novo();
                sub.compilar_bloco(corpo, None, &mut Vec::new())?;
                sub.emit(Op::ReturnNull);
                self.emit(Op::DefFunc {
                    nome: arc(""),
                    params: parametros.iter().map(|p| arc(&p.nome)).collect(),
                    corpo: sub.ops,
                });
            }

            Expressao::Lista(elementos) => {
                let n = elementos.len();
                for e in elementos {
                    self.compilar_expr(e)?;
                }
                self.emit(Op::MakeList(n));
            }

            Expressao::Mapa(pares) => {
                let chaves: Vec<Arc<str>> = pares.iter().map(|(k, _)| arc(k)).collect();
                for (_, v) in pares {
                    self.compilar_expr(v)?;
                }
                self.emit(Op::MakeMap(chaves));
            }

            Expressao::Variavel(nome) => {
                self.emit(Op::Load(arc(nome)));
            }

            Expressao::Atribuicao { nome, valor } => {
                self.compilar_expr(valor)?;
                self.emit(Op::Dup);
                self.emit(Op::Store(arc(nome)));
            }

            Expressao::AtribuicaoIndexada {
                objeto,
                indice,
                valor,
            } => {
                self.compilar_expr(objeto)?;
                self.compilar_expr(indice)?;
                self.compilar_expr(valor)?;
                self.emit(Op::SetIndex);
                if let Expressao::Variavel(nome) = objeto.as_ref() {
                    self.emit(Op::Store(arc(nome)));
                }
            }

            Expressao::UnOp { op, expr } => {
                self.compilar_expr(expr)?;
                match op {
                    OpUnario::Negativo => {
                        self.emit(Op::Neg);
                    }
                    OpUnario::Nao => {
                        self.emit(Op::Not);
                    }
                }
            }

            Expressao::BinOp { esq, op, dir } => {
                match op {
                    OpBinario::E => {
                        self.compilar_expr(esq)?;
                        self.emit(Op::Dup);
                        let jf = self.emit(Op::JumpFalse(0));
                        self.emit(Op::Pop);
                        self.compilar_expr(dir)?;
                        let fim = self.len();
                        self.patch(jf, fim);
                        return Ok(());
                    }
                    OpBinario::Ou => {
                        self.compilar_expr(esq)?;
                        self.emit(Op::Dup);
                        let jt = self.emit(Op::JumpTrue(0));
                        self.emit(Op::Pop);
                        self.compilar_expr(dir)?;
                        let fim = self.len();
                        self.patch(jt, fim);
                        return Ok(());
                    }
                    _ => {}
                }
                self.compilar_expr(esq)?;
                self.compilar_expr(dir)?;
                let op_bc = match op {
                    OpBinario::Soma => Op::Add,
                    OpBinario::Subtracao => Op::Sub,
                    OpBinario::Multiplicacao => Op::Mul,
                    OpBinario::Divisao => Op::Div,
                    OpBinario::DivisaoInteira => Op::IntDiv,
                    OpBinario::Modulo => Op::Mod,
                    OpBinario::Igual => Op::Eq,
                    OpBinario::DiferenteDe => Op::Ne,
                    OpBinario::MenorQue => Op::Lt,
                    OpBinario::MaiorQue => Op::Gt,
                    OpBinario::MenorOuIgual => Op::Le,
                    OpBinario::MaiorOuIgual => Op::Ge,
                    OpBinario::Em => {
                        self.emit(Op::Em);
                        return Ok(());
                    }
                    OpBinario::NaoEm => {
                        self.emit(Op::NaoEm);
                        return Ok(());
                    }
                    OpBinario::E | OpBinario::Ou => unreachable!(),
                };
                self.emit(op_bc);
            }

            // Reconhece chamadas a funcoes tensoriais e emite opcodes dedicados
            Expressao::ChamadaFuncao { nome, args } => {
                let n = args.len();
                for a in args {
                    self.compilar_expr(a)?;
                }
                let op = match nome.as_str() {
                    // Produto matricial / transposição
                    "tensor_matmul" | "mat_mul" if n == 2 => Op::TensorMatMul,
                    "tensor_transpor" | "mat_transpor" if n == 1 => Op::TensorTranspor,
                    // Ativações
                    "tensor_relu" if n == 1 => Op::TensorReLU,
                    "tensor_sigmoid" if n == 1 => Op::TensorSigmoid,
                    "tensor_softmax" if n == 1 => Op::TensorSoftmax,
                    "tensor_tanh" if n == 1 => Op::TensorTanh,
                    // Element-wise com dois operandos
                    "tensor_add" | "tensor_soma" if n == 2 => Op::TensorAdd,
                    "tensor_sub" if n == 2 => Op::TensorSub,
                    "tensor_mul" if n == 2 => Op::TensorMulElem,
                    "tensor_div" if n == 2 => Op::TensorDivElem,
                    "tensor_potencia" if n == 2 => Op::TensorPow,
                    // Unárias
                    "tensor_neg" if n == 1 => Op::TensorNeg,
                    "tensor_exp" if n == 1 => Op::TensorExp,
                    "tensor_log" if n == 1 => Op::TensorLog,
                    "tensor_raiz" if n == 1 => Op::TensorSqrt,
                    // Reduções globais
                    "tensor_soma_total" if n == 1 => Op::TensorSomaTotal,
                    "tensor_media" if n == 1 => Op::TensorMediaTotal,
                    "tensor_max" if n == 1 => Op::TensorMaxTotal,
                    "tensor_min" if n == 1 => Op::TensorMinTotal,
                    // Reduções por eixo
                    "tensor_soma_eixo" if n == 2 => Op::TensorSomaEixo,
                    "tensor_media_eixo" if n == 2 => Op::TensorMediaEixo,
                    "tensor_max_eixo" if n == 2 => Op::TensorMaxEixo,
                    "tensor_min_eixo" if n == 2 => Op::TensorMinEixo,
                    // Forma
                    "tensor_concatenar" if n == 2 => Op::TensorConcatenar,
                    "tensor_empilhar" if n == 2 => Op::TensorEmpilhar,
                    _ => Op::CallNative(arc(nome), n),
                };
                self.emit(op);
            }

            Expressao::Chamada { funcao, args } => {
                self.compilar_expr(funcao)?;
                let n = args.len();
                for a in args {
                    self.compilar_expr(a)?;
                }
                self.emit(Op::Call(n));
            }

            Expressao::Acesso { objeto, indice } => {
                self.compilar_expr(objeto)?;
                self.compilar_expr(indice)?;
                self.emit(Op::GetIndex);
            }

            // Null-coalescing: a ?? b → se a for nulo, avalia b
            Expressao::NullCoalescente { esq, dir } => {
                self.compilar_expr(esq)?;
                self.emit(Op::Dup);
                // Se nao for nulo, pula o dir (usa o dup)
                // Estratégia: push nulo, compara Eq, JumpFalse sobre [pop, compilar_dir]
                self.emit(Op::PushNull);
                self.emit(Op::Eq);
                let jf = self.emit(Op::JumpFalse(0)); // se a != nulo, pula
                self.emit(Op::Pop); // descarta o dup (que era nulo)
                self.compilar_expr(dir)?;
                self.patch(jf, self.len());
            }

            // Acesso opcional: a?.campo → se a for nulo retorna nulo, senao a.campo
            Expressao::AcessoOpcional { objeto, chave } => {
                self.compilar_expr(objeto)?;
                self.emit(Op::Dup);
                self.emit(Op::PushNull);
                self.emit(Op::Eq);
                let jf = self.emit(Op::JumpFalse(0)); // se nao nulo, faz o acesso
                                                      // Era nulo: pop o dup, push nulo
                                                      // (ja temos null no topo após o Dup ser comparado? não — Eq consome os dois)
                                                      // Na verdade Dup deixou uma copia, mas Eq a consumiu junto com PushNull
                                                      // O que sobrou é o Dup original. Se era nulo, ele ainda está.
                let pula_fim = self.emit(Op::Jump(0));
                let acesso_ini = self.len();
                self.patch(jf, acesso_ini);
                self.emit(Op::PushStr(chave.clone()));
                self.emit(Op::GetIndex);
                let fim = self.len();
                self.patch(pula_fim, fim);
            }

            // Funcao seta: compilada como DefFunc
            Expressao::FuncaoSeta { parametros, corpo } => {
                let mut sub = Compilador::novo();
                sub.compilar_expr(corpo)?;
                sub.emit(Op::Return);
                self.emit(Op::DefFunc {
                    nome: arc(""),
                    params: parametros.iter().map(|p| arc(p)).collect(),
                    corpo: sub.ops,
                });
            }
        }
        Ok(())
    }
}
