"""CardioIA — Fase 2, Parte 1: extração de sintomas e sugestão de diagnóstico.

Lê os relatos de pacientes (data/relatos_pacientes.txt), procura neles as
expressões do mapa de conhecimento (data/mapa_conhecimento.csv) e sugere o
diagnóstico de maior pontuação.

Uso:
    python src/extrator_diagnostico.py                       # analisa os 10 relatos
    python src/extrator_diagnostico.py "sinto falta de ar"   # analisa frases livres
    python src/extrator_diagnostico.py --demo                # auto-verificação

Uso educacional: não substitui avaliação médica.
"""
import csv
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

DADOS = Path(__file__).resolve().parent.parent / "data"

# Negação no estilo NegEx: gatilho explícito até 2 palavras antes do sintoma, na mesma oração.
NEGACAO = re.compile(r"\b(?:nao (?:sinto|senti|tenho|tive|estou com|apresento)|sem|nem)\b(?:\s+\w+){0,2}\s*$")
FIM_DE_ORACAO = re.compile(r"[,.;:!?]|\b(?:mas|porem)\b")


def normalizar(texto):
    """Minúsculas e sem acento: 'Tórax' e 'torax' viram a mesma coisa."""
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().lower()


def padrao(expressao):
    # ponytail: aceita até 2 palavras no meio ("aperto muito forte no peito" casa "aperto no peito");
    # sem lematização: plural ou conjugação nova entra como nova linha no mapa
    palavras = map(re.escape, normalizar(expressao).split())
    return re.compile(r"\b" + r"(?:\s+\w+){0,2}\s+".join(palavras) + r"\b")


def negado(texto, inicio):
    # ponytail: só nega com gatilho explícito; na dúvida o sintoma conta (em triagem, falso negativo custa mais)
    oracao = FIM_DE_ORACAO.split(texto[:inicio])[-1]
    return bool(NEGACAO.search(oracao))


def carregar_mapa(caminho=DADOS / "mapa_conhecimento.csv"):
    with open(caminho, encoding="utf-8-sig", newline="") as f:
        regras = [(l["Sintoma 1"], l["Doença Associada"],
                   [padrao(e) for e in (l["Sintoma 1"], l["Sintoma 2"]) if e.strip()])
                  for l in csv.DictReader(f)]
    # nº de doenças distintas por sintoma: exclusivo vale 1 ponto, compartilhado por k doenças vale 1/k
    especificidade = Counter(s for s, _ in {(s, d) for s, d, _ in regras})
    return regras, especificidade


def analisar(frase, regras, especificidade):
    """Devolve (sintomas identificados, sintomas negados, ranking [(doença, pontos)])."""
    texto = normalizar(frase)
    achados, negados = set(), set()
    for sintoma, doenca, padroes in regras:
        inicios = [m.start() for p in padroes for m in p.finditer(texto)]
        if any(not negado(texto, i) for i in inicios):
            achados.add((sintoma, doenca))
        elif inicios:
            negados.add(sintoma)
    pontos = Counter()
    for sintoma, doenca in achados:
        pontos[doenca] += 1 / especificidade[sintoma]
    sintomas = sorted({s for s, _ in achados})
    ranking = sorted(pontos.items(), key=lambda par: (-par[1], par[0]))
    return sintomas, sorted(negados - set(sintomas)), ranking


def main(frases):
    regras, especificidade = carregar_mapa()
    print("CardioIA: sugestão de diagnóstico (uso educacional, não substitui avaliação médica)\n")
    for n, frase in enumerate(frases, 1):
        sintomas, negados, ranking = analisar(frase, regras, especificidade)
        print(f"[{n:02d}] {frase}")
        print(f"     Sintomas identificados: {', '.join(sintomas) or 'nenhum'}")
        if negados:
            print(f"     Sintomas negados:       {', '.join(negados)}")
        if not ranking:
            print("     Diagnóstico sugerido:   nenhum sintoma do mapa reconhecido; encaminhar para avaliação\n")
            continue
        topo = [d for d, p in ranking if math.isclose(p, ranking[0][1])]
        empate = " (empate)" if len(topo) > 1 else ""
        print(f"     Diagnóstico sugerido:   {' ou '.join(topo)}{empate}, pontuação {ranking[0][1]:.2f}")
        outras = [f"{d} ({p:.2f})" for d, p in ranking if d not in topo][:3]
        if outras:
            print(f"     Outras hipóteses:       {', '.join(outras)}")
        print()


def ler_relatos(caminho=DADOS / "relatos_pacientes.txt"):
    return [l.strip() for l in caminho.read_text(encoding="utf-8-sig").splitlines() if l.strip()]


def demo():
    regras, esp = carregar_mapa()
    sintomas, negados, _ = analisar("Não sinto dor no peito, mas tenho falta de ar", regras, esp)
    assert negados == ["dor no peito"] and sintomas == ["falta de ar"], (sintomas, negados)
    assert "dor no peito" in analisar("não aguento essa dor no peito", regras, esp)[0]  # sem gatilho: não nega
    assert "dor no peito" in analisar("dor muito forte no peito", regras, esp)[0]       # janela de 2 palavras
    assert analisar("minha perna esquerda está inchada", regras, esp)[0] == ["inchaço em uma perna"]
    assert analisar("minhas pernas estão inchadas", regras, esp)[0] == ["inchaço nas pernas"]
    assert analisar("estou ótimo hoje", regras, esp) == ([], [], [])
    _, _, ranking = analisar("aperto no peito que desce para o braço, com suor frio", regras, esp)
    assert ranking[0][0] == "Infarto Agudo do Miocárdio", ranking
    print("demo ok")


if __name__ == "__main__":
    argumentos = sys.argv[1:]
    if argumentos == ["--demo"]:
        demo()
    else:
        main(argumentos or ler_relatos())
