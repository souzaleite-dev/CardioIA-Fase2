# CardioIA — Fase 2: Diagnóstico Automatizado

> Projeto acadêmico FIAP — Curso de Inteligência Artificial
> **Fase 2:** IA no estetoscópio digital. Extração de sintomas a partir de relatos de
> pacientes, sugestão de diagnóstico por mapa de conhecimento e classificação de risco
> por texto, com análise de vieses.

**Vídeo de demonstração (YouTube, não listado):** https://youtu.be/4vvgaEX1x4Q

**Fase anterior:** [CardioIA-Fase1](https://github.com/souzaleite-dev/CardioIA-Fase1) ·
**Ir Além 1 (portal React):** [souzaleite-cardioia-portal](https://github.com/souzaleite-dev/souzaleite-cardioia-portal)

---

## 1. Visão geral

| Entrega | O que faz | Resultado |
|---|---|---|
| Parte 1 — Extração de informações | Lê 10 relatos de pacientes, identifica sintomas com um mapa de conhecimento (183 linhas, 42 sintomas, 13 doenças) e sugere o diagnóstico | 10 de 10 relatos com o diagnóstico esperado, inclusive negação e infarto atípico |
| Parte 2 — Classificador de risco | Classifica frases como `alto risco` ou `baixo risco` com TF-IDF + Regressão Logística | 89,5% de acurácia no teste; análise de vieses com 12 frases-armadilha |
| Ir Além 1 — Portal | Interface em React + Vite: login simulado, pacientes, agendamento e dashboard | [repositório próprio](https://github.com/souzaleite-dev/souzaleite-cardioia-portal), como pede a atividade |
| Ir Além 2 — Rede neural em ECG | MLP em Keras que classifica imagens de batimentos (PTB) como normais ou de infarto | 95,6% de acurácia e AUC de 0,990 no teste |

Nenhum dado pessoal de saúde é usado: relatos, mapa e frases foram escritos para o projeto,
e o ECG vem de uma base pública e anonimizada.

### Mapa da entrega

| Critério de avaliação | Pontos | Onde está |
|---|---:|---|
| Relatos e mapa de conhecimento organizados | 2 | [`data/relatos_pacientes.txt`](data/relatos_pacientes.txt), [`data/mapa_conhecimento.csv`](data/mapa_conhecimento.csv) · seções 4.1 e 4.2 |
| Código de extração de informações funcional | 2 | [`src/extrator_diagnostico.py`](src/extrator_diagnostico.py) · seções 4.3 e 4.4 |
| Dataset simples criado corretamente | 1 | [`data/frases_risco.csv`](data/frases_risco.csv) · seção 5.1 |
| Classificador treinado e testado corretamente | 2 | [`notebooks/classificador_risco.ipynb`](notebooks/classificador_risco.ipynb) · seções 5.2 a 5.4 |
| Documentação clara e repositório público com README completo | 1 | este README |
| Vídeo no YouTube (não listado) com link no GitHub | 2 | link no topo deste README |
| Ir Além 1 | — | [souzaleite-cardioia-portal](https://github.com/souzaleite-dev/souzaleite-cardioia-portal) |
| Ir Além 2 | — | [`ir-alem-2/`](ir-alem-2) (README e exemplos de imagens) e [`notebooks/ir_alem_2_ecg_mlp.ipynb`](notebooks/ir_alem_2_ecg_mlp.ipynb) · seção 6.2 |

---

## 2. Estrutura do repositório

```
CardioIA-Fase2/
├── README.md
├── data/
│   ├── relatos_pacientes.txt         # Parte 1 — 10 relatos de pacientes, um por linha
│   ├── mapa_conhecimento.csv         # Parte 1 — Sintoma 1 | Sintoma 2 | Doença Associada
│   └── frases_risco.csv              # Parte 2 — 150 frases rotuladas (frase,situacao)
├── src/
│   └── extrator_diagnostico.py       # Parte 1 — extração de sintomas e sugestão de diagnóstico
├── notebooks/
│   ├── classificador_risco.ipynb     # Parte 2 — TF-IDF, classificação e avaliação
│   └── ir_alem_2_ecg_mlp.ipynb       # Ir Além 2 — MLP em imagens de ECG (Keras)
└── ir-alem-2/                        # Ir Além 2 — README explicativo
    ├── exemplos/                     # imagens de ECG normais e anormais (original e entrada da MLP)
    └── figuras/                      # resultados: pipeline, curvas, matriz de confusão, erros
```

---

## 3. Como executar

Requisitos: Python 3.10+. A Parte 1 usa só a biblioteca padrão; a Parte 2 precisa de
`pandas`, `scikit-learn` e `matplotlib`.

```bash
pip install pandas scikit-learn matplotlib notebook

# Parte 1
python src/extrator_diagnostico.py --demo        # auto-verificação da lógica
python src/extrator_diagnostico.py               # analisa os 10 relatos
python src/extrator_diagnostico.py "estou com falta de ar e o coração acelerado"   # frase livre

# Parte 2
jupyter notebook notebooks/classificador_risco.ipynb

# Ir Além 2 (o dataset, cerca de 100 MB, é baixado sozinho, sem conta no Kaggle)
pip install tensorflow kagglehub
jupyter notebook notebooks/ir_alem_2_ecg_mlp.ipynb
```

Os notebooks já estão salvos com todos os resultados. No Google Colab, a primeira célula
do notebook da Parte 2 mostra como clonar o repositório e ajustar o caminho; o do Ir Além 2
roda direto, sem ajuste.

---

## 4. Parte 1 — Relatos de sintomas e extração de informações

### 4.1 Relatos (`data/relatos_pacientes.txt`)

Dez frases, uma por linha. Cada uma traz **o que o paciente sente**, **quando começou** e
**como os sintomas afetam a rotina**, por exemplo:

> Há dois meses sinto um aperto no peito quando subo a ladeira de casa, que passa quando
> paro para descansar; por isso deixei de ir a pé para o trabalho.

Os relatos cobrem 9 doenças diferentes e foram escritos na linguagem do paciente
("batedeira", "suando frio", "vista embaçada"), não em termos técnicos. Dois foram
pensados para testar a lógica:

- **Relato 3** começa negando um sintoma ("Não sinto dor no peito, mas..."). Um buscador
  ingênuo de palavras leria "dor no peito" e erraria.
- **Relato 10** é um infarto com **apresentação atípica**: enjoo, suor frio e dor na
  mandíbula, sem dor no peito, em uma paciente diabética. É o quadro que mais se perde
  em triagem, e é mais frequente em mulheres, idosos e diabéticos.

### 4.2 Mapa de conhecimento (`data/mapa_conhecimento.csv`)

Colunas `Sintoma 1 | Sintoma 2 | Doença Associada`, com uma convenção simples:
**Sintoma 1** é o nome do sintoma e **Sintoma 2** é outra forma de o paciente dizer a
mesma coisa (sinônimo, termo popular ou técnico).

| Sintoma 1 | Sintoma 2 | Doença Associada |
|---|---|---|
| aperto no peito | peso no peito | Infarto Agudo do Miocárdio |
| palpitação | batedeira | Fibrilação Atrial |
| falta de ar ao deitar | acordo sem ar | Insuficiência Cardíaca |
| cansaço constante | fadiga | Insuficiência Cardíaca |

São **183 linhas**: 42 sintomas, cada um com suas variações, ligados a 13 doenças —
Infarto Agudo do Miocárdio, Angina Estável, Insuficiência Cardíaca, Fibrilação Atrial,
Taquicardia Supraventricular, Bradicardia, Crise Hipertensiva, AVC, Pericardite,
Miocardite, Embolia Pulmonar, Trombose Venosa Profunda e Estenose Aórtica.

A relação é **muitos-para-muitos**: "falta de ar" aparece em 6 doenças, e o mapa repete
o sintoma em cada uma. Algumas distinções clínicas ficaram codificadas:

- **aperto ou peso no peito** (dor em constrição) aponta para isquemia — infarto e
  angina; **pontada no peito** (dor que piora ao respirar) aponta para pericardite e
  embolia pulmonar;
- **pernas inchadas** (as duas) aponta para insuficiência cardíaca; **perna inchada**
  (uma só) aponta para trombose venosa profunda;
- sintomas **atípicos** de infarto (náusea, dor na mandíbula, dor na boca do estômago,
  cansaço) entraram de propósito, para não repetir o viés da descrição "clássica".

### 4.3 Lógica de extração (`src/extrator_diagnostico.py`)

1. **Normalização:** texto em minúsculas e sem acento, para "Tórax" e "torax" serem iguais.
2. **Busca das expressões do mapa**, aceitando até 2 palavras no meio: "aperto **muito
   forte** no peito" casa com "aperto no peito". A busca respeita fronteira de palavra,
   então "perna" não casa com "pernas".
3. **Negação**, no estilo do algoritmo NegEx: um sintoma precedido de gatilho explícito
   ("não sinto", "não tenho", "sem", "nem"...) na mesma oração é descartado e listado
   como negado. Só gatilhos explícitos negam: na dúvida o sintoma conta, porque em
   triagem um falso negativo custa mais que um falso positivo.
4. **Pontuação por especificidade:** um sintoma exclusivo de uma doença vale 1 ponto; um
   sintoma compartilhado por *k* doenças vale 1/*k*. É a mesma intuição do IDF da Parte 2:
   o que é raro informa mais. A doença de maior pontuação é o diagnóstico sugerido; empate
   é declarado, não escondido.

### 4.4 Resultado nos 10 relatos

| # | Sintomas identificados | Diagnóstico sugerido | Pontos |
|---:|---|---|---:|
| 1 | aperto no peito, piora com esforço, melhora com repouso | Angina Estável | 1,83 |
| 2 | aperto no peito, dor no braço esquerdo, suor frio | Infarto Agudo do Miocárdio | 2,00 |
| 3 | inchaço nas pernas, falta de ar ao deitar · *negado: dor no peito* | Insuficiência Cardíaca | 2,00 |
| 4 | palpitação, batimento irregular, tontura | Fibrilação Atrial | 1,75 |
| 5 | tontura, cansaço, desmaio, pulso lento | Bradicardia | 1,95 |
| 6 | dor de cabeça forte, dor na nuca, visão embaçada, zumbido no ouvido, pressão alta | Crise Hipertensiva | 4,00 |
| 7 | boca torta, fala enrolada, fraqueza em um lado do corpo | AVC | 3,00 |
| 8 | inchaço em uma perna, perna vermelha e quente, dor na panturrilha | Trombose Venosa Profunda | 2,50 |
| 9 | pontada no peito, dor ao respirar fundo, melhora ao inclinar para frente, após infecção viral | Pericardite | 2,50 |
| 10 | náusea, suor frio, dor na mandíbula, cansaço · *negado: dor no peito* | Infarto Agudo do Miocárdio | 2,20 |

Saída completa de um relato:

```
[10] Desde ontem à tarde sinto um enjoo que não passa, suor frio e uma dor estranha na mandíbula, sem dor no peito; sou diabética e estou tão cansada que não consegui cuidar da casa hoje.
     Sintomas identificados: cansaço, dor na mandíbula, náusea, suor frio
     Sintomas negados:       dor no peito
     Diagnóstico sugerido:   Infarto Agudo do Miocárdio, pontuação 2.20
     Outras hipóteses:       Angina Estável (0.50), Crise Hipertensiva (0.50), Bradicardia (0.20)
```

Toda sugestão vem com os sintomas que a justificam, então qualquer pessoa consegue
auditar por que o sistema chegou àquela hipótese.

**Limitações conhecidas:** não há lematização (plural ou conjugação nova entra como nova
linha no mapa); a janela de 2 palavras é fixa; a negação depende de uma lista curta de
gatilhos; a pontuação ordena hipóteses, mas não é probabilidade; e o mapa foi montado
a partir de literatura, sem validação por profissional de saúde.

---

## 5. Parte 2 — Classificador de risco por texto

Detalhes, gráficos e código em [`notebooks/classificador_risco.ipynb`](notebooks/classificador_risco.ipynb).

### 5.1 Base rotulada (`data/frases_risco.csv`)

150 frases curtas no formato `frase,situacao`, balanceadas (75 de alto risco, 75 de baixo
risco). O critério de rotulagem segue os sinais de alarme usados em triagem
(discriminadores das cores vermelha e laranja do Protocolo de Manchester e sinais de
alerta de infarto e AVC):

| alto risco | baixo risco |
|---|---|
| dor ou aperto no peito em repouso, intenso, irradiado ou com suor frio, náusea ou falta de ar | dor localizada que piora ao toque ou ao movimento, com causa evidente |
| falta de ar em repouso, ao deitar ou de início súbito | cansaço com causa evidente |
| desmaio, quase desmaio, palpitação com tontura | palpitação curta depois de café, susto ou nervosismo |
| boca torta, fala enrolada, fraqueza ou dormência súbita de um lado | queixas leves de outros sistemas (resfriado, azia, dente) |
| pressão muito alta com sintomas, tosse com sangue | rotina: receita, check-up, pressão controlada |

Cerca de 30 das 75 frases de baixo risco usam vocabulário cardíaco benigno ("o coração
acelerou depois que tomei três cafés"). Sem isso, o modelo separaria as classes pelo tema
da frase e a acurácia sairia inflada.

### 5.2 Modelo

- **TF-IDF** com `strip_accents='unicode'` e **sem remoção de stopwords**: as listas
  prontas em português removem "não" e "sem", o que apagaria a negação.
- Separação 75% treino / 25% teste, estratificada, antes de qualquer ajuste.
- Escolha do modelo por validação cruzada estratificada de 5 dobras no treino, com o
  TF-IDF dentro do `Pipeline` para não vazar vocabulário entre dobras:

| Modelo | Acurácia | Sensibilidade (alto risco) |
|---|---|---|
| **Regressão Logística · palavras** | **0,83 ± 0,06** | **0,89 ± 0,04** |
| Regressão Logística · palavras + bigramas | 0,80 ± 0,07 | 0,87 ± 0,09 |
| Árvore de Decisão · palavras | 0,76 ± 0,15 | 0,75 ± 0,14 |

A Regressão Logística venceu e tem uma vantagem extra em saúde: cada palavra recebe um
peso legível, o que permite auditar o que o modelo aprendeu.

### 5.3 Resultados no teste (38 frases)

| Limiar de decisão | Acurácia | Sensibilidade (alto risco) | Especificidade (baixo risco) |
|---|---:|---:|---:|
| 0,50 (padrão) | 89,5% | 0,89 | 0,89 |
| **0,45 (adotado)** | 81,6% | **0,95** | 0,68 |

Em triagem, deixar passar um caso grave custa mais que um alarme falso. Por isso o limiar
foi baixado para 0,45, valor escolhido com validação cruzada **no treino** (sensibilidade
de 0,89 para 0,98 sem perder acurácia), sem olhar o teste. No teste, só 1 dos 19 casos
graves escapa; em troca, os alarmes falsos sobem de 2 para 6.

### 5.4 Vieses e distorções encontrados

A acurácia de 89,5% esconde problemas que só aparecem olhando os pesos do modelo e
testando frases fora da base:

- **Negação invertida.** No treino, "não" aparece em 6 frases graves ("não passa",
  "não consigo respirar") e em nenhuma leve, e virou um dos termos mais fortes de alto
  risco. Resultado: "não sinto dor no peito nem falta de ar" recebe P(alto) = 0,67.
  TF-IDF é um saco de palavras e não entende negação; a Parte 1, com regra explícita,
  entende.
- **Viés do rotulador.** "está", "para" e "minha" estão entre os termos que mais empurram
  para alto risco, e "depois" (19 frases leves, 2 graves no treino) é o que mais empurra
  para baixo. O modelo aprendeu o estilo de quem escreveu a base. Foi isso que fez
  "dor no peito e tosse com sangue depois de uma cirurgia", quadro sugestivo de embolia
  pulmonar, cair em baixo risco no teste: trocando só "depois de" por "após", P(alto) vai
  de 0,49 para 0,58 e a frase passa a alto risco.
- **Infarto atípico na fronteira.** "sinto enjoo, suor frio e uma dor estranha na
  mandíbula" ficou com P(alto) = 0,50: com o limiar padrão, seria baixo risco. A base
  retrata o infarto clássico (dor no peito irradiando para o braço), descrição construída
  sobre coortes majoritariamente masculinas. É o mesmo viés de sexo apontado na Fase 1.
- **Linguagem do paciente.** Gíria, regionalismo, jargão e erro de digitação viram
  palavras desconhecidas. Uma frase só com palavras fora do vocabulário recebe
  P(alto) ≈ 0,48, praticamente cara ou coroa. O viés atinge quem fala diferente de quem
  escreveu a base: outra região, outra escolaridade.
- **Acertos pelo motivo errado.** Em "precordialgia com irradiação para membro superior
  esquerdo e sudorese", 5 dos 8 termos são desconhecidos; o alto risco veio de "para",
  "com" e "esquerdo".

Nas 12 frases-armadilha, o modelo acertou 9; os 3 erros são alarmes falsos e nenhum caso
grave passou despercebido. Mitigações discutidas no notebook: marcar negação no
pré-processamento, base com frases de pacientes reais de várias regiões rotuladas por
profissionais, inclusão de apresentações atípicas com sensibilidade medida por subgrupo e
modelos que leem contexto, como o BERTimbau.

---

## 6. Ir Além

### 6.1 Ir Além 1 — Portal do CardioIA (React + Vite)

Repositório próprio, como pede a atividade:
[souzaleite-cardioia-portal](https://github.com/souzaleite-dev/souzaleite-cardioia-portal).
Login simulado com JWT fake e Context API, rotas protegidas, lista de pacientes (30 da base
sintética da Fase 1), agendamento com `useReducer` e validação de conflitos, dashboard e
CSS Modules responsivo. Instruções, telas e decisões técnicas estão no README de lá.

**Vídeo do portal (YouTube, não listado):** https://youtu.be/EK6RL6GIeNY

### 6.2 Ir Além 2 — Diagnóstico visual de ECG com rede neural MLP

Notebook: [`notebooks/ir_alem_2_ecg_mlp.ipynb`](notebooks/ir_alem_2_ecg_mlp.ipynb) · README explicativo com
exemplos de imagens: [`ir-alem-2/README.md`](ir-alem-2/README.md).

- **Dados:** PTB Diagnostic ECG Database, na versão do Kaggle recomendada pela atividade
  (`shayanfazeli/heartbeat`): 14.552 batimentos da derivação II, 4.046 normais (controles
  saudáveis) e 10.506 anormais (infarto do miocárdio).
- **Imagens:** o dataset traz cada batimento como 187 números; cada um foi desenhado como
  traçado sobre papel de ECG (128 × 256, colorido) e pré-processado como uma foto de exame:
  tons de cinza, inversão, redimensionamento para 32 × 64 e normalização, num vetor de
  2.048 pixels.
- **MLP (Keras):** 2.048 → 256 → 128 → 1, com *dropout*, pesos por classe para o
  desbalanceamento e parada antecipada. Divisão estratificada 70/15/15.

| No teste (2.183 batimentos) | Acurácia | Sensibilidade | Especificidade | AUC |
|---|---:|---:|---:|---:|
| **MLP sobre imagens** | **95,6%** | **98,2%** | 89,1% | **0,990** |
| MLP sobre o sinal bruto (187 amostras) | 94,9% | 95,1% | 94,2% | 0,986 |
| Referência: responder sempre "anormal" | 72,2% | 100% | 0% | 0,5 |

**O achado mais importante:** deslocar a imagem **1 pixel** para o lado derruba a acurácia
para 75,4%, e 8 pixels levam a AUC a 0,509, nível de acaso. A MLP decorou em que pixel cada
parte do traçado costuma estar, em vez de aprender a forma do batimento; só funciona porque
todas as imagens foram geradas alinhadas. É o argumento para as redes convolucionais da
Fase 4. O notebook também discute o vazamento entre pacientes (o CSV não identifica a
pessoa de cada batimento) e o desequilíbrio de sexo da base PTB (72% de homens).

---

## 7. Governança, ética e responsabilidade

- **Privacidade.** Relatos, mapa e frases foram escritos para o projeto; não há dado de
  paciente real, nome, documento ou qualquer identificador. Sem titular de dados, a LGPD
  não se aplica, e o repositório pode ser público, mesmo princípio da Fase 1. O Ir Além 2
  usa ECGs reais, mas de uma base pública e anonimizada (PTB, via PhysioNet), baixada na
  hora da execução e não copiada para o repositório.
- **Rastreabilidade.** O critério de rotulagem está documentado, a separação treino/teste
  tem semente fixa (`random_state=42`) e o notebook reproduz os mesmos números a cada
  execução.
- **Limites da rotulagem.** Os rótulos seguem um critério escrito, não a avaliação de um
  profissional de saúde, e foram atribuídos por uma única pessoa, o que é exatamente o
  viés do rotulador observado na seção 5.4.
- **Responsabilidade no uso.** As duas soluções são **apoio**, não decisão: sugerem uma
  hipótese ou uma prioridade e mostram o porquê (sintomas encontrados na Parte 1, pesos
  das palavras na Parte 2). Em uso real, o profissional de saúde decide, o limiar prioriza
  sensibilidade e o sistema precisaria de validação clínica e de regularização como
  software dispositivo médico junto à ANVISA (RDC nº 657/2022).
- **Uso exclusivamente educacional.** Nada aqui deve apoiar decisão sobre paciente real.

---

## 8. Conexão com a Fase 1

- O diagnóstico de vieses da Fase 1 (base numérica com 61% de homens; mulheres com
  sintomas atípicos são subdiagnosticadas) virou **teste concreto** aqui: o relato 10 da
  Parte 1 e a frase-armadilha de infarto atípico da Parte 2.
- As classes do acervo de ECG da Fase 1 (taquicardia, bradicardia, fibrilação, isquemia)
  aparecem como doenças do mapa de conhecimento: Taquicardia Supraventricular,
  Bradicardia, Fibrilação Atrial, Infarto e Angina.
- As variáveis de sintomas da base numérica (`dor_toracica`, `angina_esforco`,
  `dispneia`) têm correspondentes no mapa: "aperto no peito", "piora com esforço",
  "falta de ar".
- O princípio de dados se mantém: nenhum dado pessoal, tudo documentado e reproduzível.

---

## 9. Referências

- Mackway-Jones K, Marsden J, Windle J (eds.). *Emergency Triage: Manchester Triage Group*. 3rd ed. Wiley-Blackwell; 2014.
- Chapman WW et al. A simple algorithm for identifying negated findings and diseases in discharge summaries (NegEx). *J Biomed Inform.* 2001;34(5):301–310.
- Mehta LS et al. Acute Myocardial Infarction in Women: A Scientific Statement From the American Heart Association. *Circulation.* 2016;133(9):916–947.
- Nicolau JC et al. Diretrizes da Sociedade Brasileira de Cardiologia sobre Angina Instável e Infarto Agudo do Miocárdio sem Supradesnível do Segmento ST – 2021. *Arq Bras Cardiol.* 2021;117(1):181–264.
- Geirhos R et al. Shortcut learning in deep neural networks. *Nature Machine Intelligence.* 2020;2:665–673.
- Souza F, Nogueira R, Lotufo R. BERTimbau: Pretrained BERT Models for Brazilian Portuguese. *BRACIS*, 2020.
- scikit-learn — `TfidfVectorizer` e `LogisticRegression`: https://scikit-learn.org
- Kachuee M, Fazeli S, Sarrafzadeh M. ECG Heartbeat Classification: A Deep Transferable Representation. *IEEE ICHI*, 2018. arXiv:1805.00794. Dataset: https://www.kaggle.com/datasets/shayanfazeli/heartbeat
- Goldberger AL et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation.* 2000;101(23):e215–e220. PTB Diagnostic ECG Database: https://physionet.org/content/ptbdb/1.0.0/
- de Chazal P, O'Dwyer M, Reilly RB. Automatic classification of heartbeats using ECG morphology and heartbeat interval features. *IEEE Trans Biomed Eng.* 2004;51(7):1196–1206.
- ANVISA. Resolução RDC nº 657, de 24 de março de 2022 — regularização de software como dispositivo médico.
- Brasil. Lei nº 13.709/2018 (Lei Geral de Proteção de Dados Pessoais). https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm

---

**Integrante:** Bruno de Souza Leite — RM 567213 · trabalho individual
**FIAP — Inteligência Artificial** · Fase 2 — Diagnóstico Automatizado · 2026
