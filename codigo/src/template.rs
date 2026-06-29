use crate::ast::Programa;
use crate::interpretador::Interpretador;
/// Processador de templates PEP  -  suporta codigo embutido com <?pep ?> e <?= ?>
///
/// Estrategia: converte o template inteiro em um unico programa PEP antes de executar.
/// Isso permite que blocos (se, enquanto, para, funcao) atravessem multiplas tags.
///
/// Texto literal  ->  escrever("...")
/// <?pep ... ?>   ->  codigo PEP inserido diretamente
/// <?= expr ?>    ->  escrever(expr)
use crate::lexer::Lexer;
use crate::parser::Parser;

#[derive(Debug)]
enum Segmento<'a> {
    Texto(&'a str),
    Codigo(&'a str),
    Expressao(&'a str),
}

#[derive(Copy, Clone)]
enum TipoTag {
    CodigoLongo,
    CodigoCurto,
    Expressao,
}

fn encontrar_tag_curta(s: &str) -> Option<usize> {
    let mut inicio = 0;
    while let Some(pos_rel) = s[inicio..].find("<?") {
        let pos = inicio + pos_rel;
        if !s[pos..].starts_with("<?=") && !s[pos..].starts_with("<?pep") {
            return Some(pos);
        }
        inicio = pos + 2;
    }
    None
}

fn segmentar(fonte: &str) -> Result<Vec<Segmento<'_>>, String> {
    let mut segmentos = Vec::new();
    let mut resto = fonte;
    let mut offset_linha = 1usize;

    while !resto.is_empty() {
        let pos_pep = resto.find("<?pep");
        let pos_eq = resto.find("<?=");
        let pos_curta = encontrar_tag_curta(resto);

        let mut candidatos: Vec<(usize, TipoTag)> = Vec::new();
        if let Some(p) = pos_pep {
            candidatos.push((p, TipoTag::CodigoLongo));
        }
        if let Some(p) = pos_eq {
            candidatos.push((p, TipoTag::Expressao));
        }
        if let Some(p) = pos_curta {
            candidatos.push((p, TipoTag::CodigoCurto));
        }
        candidatos.sort_by_key(|(pos, _)| *pos);
        let proximo = candidatos.first().copied();

        match proximo {
            None => {
                if !resto.is_empty() {
                    segmentos.push(Segmento::Texto(resto));
                }
                break;
            }
            Some((pos, tipo)) => {
                if pos > 0 {
                    segmentos.push(Segmento::Texto(&resto[..pos]));
                    offset_linha += conta_linhas(&resto[..pos]);
                }
                let linha_tag = offset_linha;

                match tipo {
                    TipoTag::Expressao => {
                        let after = &resto[pos + 3..]; // pula "<?="
                        match after.find("?>") {
                            None => {
                                return Err(format!("Tag '<?=' nao fechada (linha {})", linha_tag))
                            }
                            Some(fim) => {
                                segmentos.push(Segmento::Expressao(after[..fim].trim()));
                                offset_linha += conta_linhas(&after[..fim]);
                                resto = &after[fim + 2..];
                            }
                        }
                    }
                    TipoTag::CodigoLongo | TipoTag::CodigoCurto => {
                        let inicio_codigo = if matches!(tipo, TipoTag::CodigoLongo) {
                            pos + 5
                        } else {
                            pos + 2
                        };
                        let after = &resto[inicio_codigo..];
                        match after.find("?>") {
                            None => {
                                return Err(format!(
                                    "Tag de codigo PEP nao fechada (linha {})",
                                    linha_tag
                                ))
                            }
                            Some(fim) => {
                                segmentos.push(Segmento::Codigo(&after[..fim]));
                                offset_linha += conta_linhas(&after[..fim]);
                                resto = &after[fim + 2..];
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(segmentos)
}

fn conta_linhas(s: &str) -> usize {
    s.chars().filter(|&c| c == '\n').count()
}

/// Escapa uma string de texto para uso em literal PEP: escapa \ e "
fn escapar(texto: &str) -> String {
    texto
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('{', "{{")
        .replace('}', "}}")
}

fn converter_para_programa(fonte: &str) -> Result<String, String> {
    let segmentos = segmentar(fonte)?;
    let mut programa = String::new();

    for segmento in &segmentos {
        match segmento {
            Segmento::Texto(t) => {
                if !t.is_empty() {
                    programa.push_str(&format!("escrever(\"{}\")\n", escapar(t)));
                }
            }
            Segmento::Codigo(c) => {
                programa.push_str(c);
                programa.push('\n');
            }
            Segmento::Expressao(e) => {
                programa.push_str(&format!("escrever(html_escapar(texto({})))\n", e));
            }
        }
    }

    Ok(programa)
}

pub fn verificar(fonte: &str) -> Result<(), String> {
    compilar(fonte)?;
    Ok(())
}

pub fn compilar(fonte: &str) -> Result<Programa, String> {
    let programa = converter_para_programa(fonte)?;
    let mut lexer = Lexer::novo(&programa);
    let tokens = lexer
        .tokenizar()
        .map_err(|e| format!("Erro de sintaxe no template: {}", e))?;
    let mut parser = Parser::novo(tokens);
    parser
        .parsear()
        .map_err(|e| format!("Erro de sintaxe no template: {}", e))
}

pub fn processar(fonte: &str, interp: &mut Interpretador) -> Result<(), String> {
    let ast = compilar(fonte)?;
    interp.executar(&ast)
}
