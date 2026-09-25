# Databricks notebook source
# MAGIC %md
# MAGIC #####                           **UNIVERSIDADE DE SÃO PAULO**
# MAGIC  
# MAGIC #####                                **MBA USP ESALQ**		
# MAGIC 		
# MAGIC **Estruturação metodológica para apoio à mediação de conflitos socioambientais**	
# MAGIC 	
# MAGIC Aluna: Leticia Pereira Lira
# MAGIC
# MAGIC Orientadora: Samira Sestari Nascimento 		
# MAGIC 		
# MAGIC 2026
# MAGIC
# MAGIC -----------------------------
# MAGIC Trabalho de Conclusão de Curso apresentado para obtenção do título de especialista em Gestão de Projetos
# MAGIC
# MAGIC -----------------------------
# MAGIC
# MAGIC Detalhamento metodológico e resultados referentes aos módulos:
# MAGIC - Módulo II: Avaliação de Risco Ambiental
# MAGIC - Módulo III: Planejamento da Mitigação do Risco
# MAGIC
# MAGIC ------------------------------

# COMMAND ----------

# DBTITLE 1,Biblioteca
# MAGIC %pip install numpy scikit-fuzzy networkx
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Imports
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns


# COMMAND ----------

# MAGIC %md
# MAGIC #### **NÍVEL DE IMPACTO**
# MAGIC
# MAGIC Nível de impacto do risco (I): representa a magnitude das consequências do risco (Sanchéz, 2013), que será definida pela relação entre as variáveis de Toxicidade do agrotóxico (intensidade do dano ao organismo) e de reversibilidade do agravo (capacidade de resposta do organismo ao dano) sobre a saúde humana (efeito) num determinado período.
# MAGIC
# MAGIC **Conceitos Básicos:**
# MAGIC
# MAGIC - O **risco** é a contaminação por agrotóxicos.
# MAGIC
# MAGIC - As **consequências ou efeitos do risco** são as doenças provocadas por ele.
# MAGIC
# MAGIC - A **Toxicidade** é a capacidade intrínseca de uma substância química ou agente físico de produzir um efeito nocivo e causar danos a um organismo vivo.
# MAGIC
# MAGIC - O **Reversibilidade do Agravo** indica o potencial de reversibilidade do dano sobre a saúde humana, relacionado a temporalidade dos sintomas, categorizado por reversível, controlável e irreversível.
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ##### **AGRAVOS DE SAÚDE**
# MAGIC
# MAGIC Os agravos de saúde são ilustrativos e foram selecionados apenas para demonstrar os diferentes níveis de reversibilidade dos danos. Abaixo tem-se a classificação dos agravos de acordo com as diretrizes do Ministério da Saúde [MS] (2018) e dos critérios do “National Cancer Institute” [NCI] (2025).
# MAGIC
# MAGIC ----------------------------------------
# MAGIC
# MAGIC Para o Ministério da Saúde (Portaria nº 79, de 14 de dezembro de 2018) a classificação para agravos de saúde  é:
# MAGIC - Leve: "Sinais ou Sintomas leves, autoresolutivos ou transitórios"
# MAGIC - Moderado: "Sinais ou sintomas pronunciados ou prolongados"
# MAGIC - Grave: "Sinais e sintomas que ameaçam a vida do paciente"
# MAGIC
# MAGIC Para o CTCAE a classificação para agravos de saúde  é:
# MAGIC - Leve: Assintomático ou sintomas leves; apenas observações clínicas ou diagnósticas; intervenção não indicada
# MAGIC - Moderado: Intervenção mínima, local ou não invasiva indicada; limitação nas atividades instrumentais de vida diária (AIVD) ou impacto leve/moderado nas atividades diárias normais adequadas à idade (pediatria)*
# MAGIC - Grave: Clinicamente significativo, mas sem risco imediato de morte; hospitalização ou prolongamento da hospitalização indicados; incapacitante; limitação nas atividades de autocuidado (AVD) ou impacto grave nas atividades diárias normais adequadas à idade (pediatria)**
# MAGIC - Risco de Morte: Consequências com risco de morte; intervenção urgente indicada.
# MAGIC - Morte relacionada ao evento adverso (EA).
# MAGIC
# MAGIC ------------------------------------------
# MAGIC
# MAGIC **1 Intoxicação Exógena**
# MAGIC
# MAGIC **Portaria nº 79, de 14 de dezembro de 2018 - Ministério da Saúde**
# MAGIC
# MAGIC Leve:
# MAGIC - Vômito
# MAGIC - Irritação na pele e nos olhos
# MAGIC - Falta de ar, dispneia leve
# MAGIC - Vertigem
# MAGIC
# MAGIC Valida os sintomas imediatos e agudos (gastrointestinais, respiratórios, neurológicos agudos) e o protocolo de reversão do quadro via suporte médico e antídotos.
# MAGIC
# MAGIC **National Cancer Institute CTCAE (Common Terminology Criteria for Adverse Events**)
# MAGIC
# MAGIC Leve e Moderado:
# MAGIC - Define que esses graus exigem apenas intervenção mínima, não invasiva, ou tratamento de suporte imediato. Cessada a exposição e aplicado o manejo, o quadro clínico é revertido e o paciente retorna ao estado basal sem sequelas permanentes.
# MAGIC
# MAGIC **2 Síndrome Metabólica**
# MAGIC
# MAGIC **Portaria nº 79, de 14 de dezembro de 2018 - Ministério da Saúde**
# MAGIC
# MAGIC Leve:
# MAGIC - Hipoglicemia/ hipertensão leves e transitórias
# MAGIC - Hipoglicemia discreta (~ 50-70 mg / dl ou 2,8-3,9 mmol / l em adultos)
# MAGIC - Aumento discreto de enzimas séricas (ASAT, ALT ~ 2-5 x normal)
# MAGIC
# MAGIC Moderado: 
# MAGIC - Hipertensão mais pronunciada
# MAGIC - Hipoglicemia mais pronunciada (~ 30-50 mg / dl ou 1,7-2,8 mmol / l em adultos)
# MAGIC - Aumento das enzimas séricas (ASAT, ALT~ 5-50 x normal)
# MAGIC
# MAGIC Aponta a necessidade de monitoramento de parâmetros bioquímicos, hepáticos e metabólicos sistêmicos decorrentes da exposição crônica ou subcrônica.
# MAGIC
# MAGIC **National Cancer Institute CTCAE (Common Terminology Criteria for Adverse Events**)
# MAGIC
# MAGIC Moderado e Grave:
# MAGIC - Define o Grau 3 nesta categoria como uma condição que requer intervenção médica ou terapêutica contínua (uso de medicação de uso contínuo para controle glicêmico ou pressórico) para evitar complicações graves. Isso conceitua perfeitamente o caráter "controlável": a patologia é crônica, mas manejável.
# MAGIC
# MAGIC **3 Doenças do sistema nervoso - Alzheimer**
# MAGIC
# MAGIC **Portaria nº 79, de 14 de dezembro de 2018 - Ministério da Saúde**
# MAGIC
# MAGIC Leve:
# MAGIC - Sonolência, ataxia, inquietação
# MAGIC - Distúrbios visuais
# MAGIC
# MAGIC Moderado:
# MAGIC - Confusão, agitação, alucinações, delírio
# MAGIC - Sintomas extrapiramidais pronunciados
# MAGIC - Convulsões infrequentes, generalizadas ou locais
# MAGIC - Distúrbios visuais
# MAGIC
# MAGIC Grave:
# MAGIC - Paralisia afetando funções vitais
# MAGIC - Cegueira
# MAGIC
# MAGIC Valida os efeitos tardios a longo prazo. Ela cita expressamente o Transtorno Neuropsiquiátrico Crônico (COPIND), caracterizado por danos ao Sistema Nervoso Central, perda de memória, deficits cognitivos e sintomas demenciais secundários à exposição crônica.
# MAGIC
# MAGIC **National Cancer Institute CTCAE (Common Terminology Criteria for Adverse Events**)
# MAGIC
# MAGIC Grave e Risco de Morte:
# MAGIC - Estabelece que, nestes graus, o declínio cognitivo causa "limitação severa das atividades de vida diária" e "necessidade de supervisão ou cuidados 24 horas". Como o Alzheimer envolve a perda estrutural e progressiva de neurônios, o dano é clinicamente classificado como irreversível.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### **MATRIZ REVERSIBILIDADE X TOXICIDADE**
# MAGIC
# MAGIC É uma matriz de decisão determinística com sistema de pontuação que relaciona a classificação toxicológica de agrotóxicos da ANVISA (2019) com o potencial de reversibilidade do agravo de saúde a partir dos sintomas da doença, baseado em diretrizes do Ministério da Saúde [MS] (2018) e nos critérios do “National Cancer Institute” [NCI] (2025).
# MAGIC
# MAGIC ------------------------------
# MAGIC
# MAGIC **Categoria de Toxicidade** (ANVISA - RESOLUÇÃO DA DIRETORIA COLEGIADA - RDC Nº 294, DE 29 DE JULHO DE 2019)
# MAGIC - 1	Produto Extremamente Tóxico - faixa vermelha
# MAGIC - 2	Produto Altamente Tóxico - faixa vermelha
# MAGIC - 3	Produto Moderadamente Tóxico - faixa amarela
# MAGIC - 4	Produto Pouco Tóxico - faixa azul
# MAGIC - 5	Produto Improvável de Causar Dano Agudo - faixa azul
# MAGIC
# MAGIC
# MAGIC **Classificação de Reversibilidade** (Premissas adotodas com base na combinação da Portaria nº 79, de 14 de dezembro de 2018 - Ministério da Saúde - e da CTCAE (Common Terminology Criteria for Adverse Events) - National Cancer Institute)
# MAGIC - 1	Reversível: Efeito tratável
# MAGIC - 2	Controlável: Efeito de manejo continuado
# MAGIC - 3	Irreversível: Efeito com desfecho tardio grave
# MAGIC
# MAGIC **Classificação do Nível de Impacto** (Premissas)
# MAGIC - 1 Muito Baixo: Cenário de menor vulnerabilidade para a saúde humana, sendo os danos estritamente temporários e superficiais.
# MAGIC - 2 Baixo: Cenário que causa desconforto ou exije atendimento médico. inicial, não ameaçam a vida e possuem prognóstico favorável a curto prazo
# MAGIC - 3 Moderado: Cenário que começa a consolidar um prejuízo persistente à saúde humana, exigindo transição no modelo de cuidado médico.
# MAGIC - 4 Alto: Cenário de elevada gravidade e severidade, gerando um estrago profundo, crônico e de difícil controle, reduzindo significativamente a capacidade funcional e a qualidade de vida do indivíduo.
# MAGIC - 5 Crítico: Cenário de máximo perigo, sendo catastrófico e terminal para a saúde humana, onde há um comprometimento estrutural e progressivo de sistemas vitais.
# MAGIC
# MAGIC Tabela da Matriz Reversibilidade e Toxicidade (com nível de impacto)
# MAGIC
# MAGIC ![image_1790104406864.png](./image_1790104406864.png "image_1790104406864.png")
# MAGIC

# COMMAND ----------

# DBTITLE 1,Figura da Matriz Reversibilidade e Toxicidade
# Dados da matriz
matriz = {
    'Reversibilidade': ['Reversível', 'Controlável', 'Irreversível'],
    'Improvável\nDano Agudo': [1, 3, 4],
    'Pouco\nTóxico': [2, 3, 4],
    'Moderadamente\nTóxico': [2, 4, 4],
    'Altamente\nTóxico': [3, 4, 5],
    'Extremamente\nTóxico': [4, 5, 5]
}

df = pd.DataFrame(matriz).set_index('Reversibilidade')

plt.figure(figsize=(8, 4))
ax = sns.heatmap(df, annot=True, fmt='d', cmap='RdYlGn_r', cbar_kws={'label': 'Nível de Impacto'}, annot_kws={"fontsize":13})
plt.gca().collections[0].colorbar.set_label('Nível de Impacto', fontsize=12, fontweight='bold')
plt.xlabel('TOXICIDADE', fontsize=14, fontweight='bold')
plt.ylabel('REVERSIBILIDADE', fontsize=14, fontweight='bold')
plt.xticks(rotation=0, ha='center')

# Mantém apenas números inteiros na barra de heatmap
colorbar = plt.gca().collections[0].colorbar
colorbar.set_ticks([1, 2, 3, 4, 5])
colorbar.set_ticklabels([1, 2, 3, 4, 5])

# Adiciona textos nos quadrantes solicitados
# Posições: (linha, coluna)
# Reversível & Improvável Dano Agudo: (0, 0)
ax.text(0 + 0.5, 0 + 0.95, "Intoxicação\nExógena", ha='center', va='bottom', fontsize=10, color='white')

# Controlável & Improvável Dano Agudo: (1, 0)
ax.text(0 + 0.5, 1 + 0.95, "Síndrome\nMetabólica", ha='center', va='bottom', fontsize=10, color='black')

# Irreversível & Improvável Dano Agudo: (2, 0)
ax.text(0 + 0.5, 2 + 0.95, "Alzheimer", ha='center', va='bottom', fontsize=10, color='white')

plt.tight_layout()
plt.savefig('matriz_impacto.png', dpi=300)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### **NÍVEL DE RISCO AMBIENTAL**
# MAGIC
# MAGIC Nível de Risco Ambiental (R): é a multiplicação do nível de exposição ao risco (E) pelo nível de impacto (I),  sendo uma adaptação simplificada da probabilidade da ocorrência de um evento pela magnitude das consequencias de Sanchéz, (2013) e da probabilidade de ocorrência e seu efeito em potencial do PMBOK (PMI, 2017).
# MAGIC
# MAGIC ![image_1790167776814.png](./image_1790167776814.png "image_1790167776814.png")
# MAGIC
# MAGIC em que I é o nível de impacto e E é o nível de exposição.
# MAGIC
# MAGIC **Classificação do Nível de Risco** (Premissas):
# MAGIC - 0-4,9: Muito Baixo 
# MAGIC - 5 - 9,9: Baixo 
# MAGIC - 10 - 14,9: Moderado 
# MAGIC - 15 - 19,9: Alto 
# MAGIC - 20 - 25: Crítico
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ##### **NÍVEL DE EXPOSIÇÃO**
# MAGIC
# MAGIC Nível de exposição ao risco (E): é estimado de forma simplificada para medir a possibilidade de ocorrência do risco.
# MAGIC
# MAGIC **Referência**
# MAGIC
# MAGIC A Hybrid Modeling Approach for Estimating the Exposure to Organophosphate Pesticide Drift in Sangamon County, Illinois (Afandi et al., 2024):
# MAGIC - Até 400 metros tem-se a maior concentração de produto da deriva
# MAGIC - Até 2000 metros há a redução até não haver mais concentração de produto da deriva
# MAGIC
# MAGIC **--Premissas**
# MAGIC - O nível de exposição ao risco será estimado exclusivamente com base na distância da área de plantio até a área de risco de contaminação, sendo utilizado como proxy da exposição na ausência de dados detalhados sobre vias de transporte dos agrotóxicos (como direção de ventos, topografia e permeabilidade do solo);
# MAGIC - Assume-se, considerando a literatura sobre deriva aérea (Contiero et al., 2018), que quanto menor a distância, maior a concentração e a chance de contato com o produto da deriva, e vice-versa
# MAGIC
# MAGIC **Classificação do Probabilidade de exposição** (Premissas):
# MAGIC - 1 Muito Baixo (acima de 2000m) 
# MAGIC - 2 Baixo (entre 1200m e 2000m) 
# MAGIC - 3 Moderado (entre 400m e 1200m)
# MAGIC - 4 Alto (entre 200m e 400m)
# MAGIC - 5 Crítico (até 200m) 
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Cálculo do Nível de Risco
# Definição dos níveis de impacto e exposição
niveis = [1, 2, 3, 4, 5]

# Criação da matriz de produtos (R = I × E)
R = np.outer(niveis, niveis)  # linhas = Impacto, colunas = Exposição

# DataFrame para visualização com colunas invertidas
df = pd.DataFrame(R, 
                  index=[i for i in niveis],
                  columns=[i for i in niveis])

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### **MATRIZ EXPOSIÇÃO E IMPACTO**
# MAGIC
# MAGIC Matriz criada a partir multiplicação do nível de impacto do risco pelo nível de exposição, para determinar o nível de do risco, sendo uma representação adapatada da matriz de probabilidade e impacto do PMBOK (PMI, 2017).

# COMMAND ----------

# DBTITLE 1,Figura da Matriz de Exposição e Impacto
# Definição das classes do risco
def classificar_risco(valor):
    if valor < 5:
        return 'Muito Baixo'
    elif valor < 10:
        return 'Baixo'
    elif valor < 15:
        return 'Moderado'
    elif valor < 20:
        return 'Alto'
    else:
        return 'Crítico'

# Aplicar classificação a cada célula
classificacao = np.vectorize(classificar_risco)(R)

# Criar anotações apenas com o valor
annotations = np.empty(R.shape, dtype=object)
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        annotations[i, j] = f"{R[i, j]}"

# Plot do heatmap com números nos quadrantes
plt.figure(figsize=(9, 7))
ax = sns.heatmap(df, fmt='', cmap='RdYlGn_r', 
                 annot=annotations,  # insere números nos quadrantes
                 annot_kws={"fontsize":13},  # aumenta fonte dos quadrantes
                 cbar_kws={'label': 'Nível de Risco', 'ticks': [1, 5, 10, 15, 20, 25]},
                 square=True, linewidths=0, linecolor='gray')

plt.xlabel('NÍVEL DE EXPOSIÇÃO', fontsize=14, fontweight='bold')
plt.ylabel('NÍVEL DE IMPACTO', fontsize=14, fontweight='bold')
plt.yticks(rotation=360)  # rotaciona os números da legenda do eixo vertical

# Aumenta o tamanho e coloca em negrito a legenda do colorbar
cbar = ax.collections[0].colorbar
cbar.set_label('Nível de Risco', fontsize=12, fontweight='bold')

# Adiciona nomenclaturas nos quadrantes solicitados
# Impacto = linhas, Exposição = colunas
# Impacto 1, Exposição 5: Intoxicação Exógena
ax.text(4 + 0.5, 0 + 0.65, "Intoxicação\nExógena", ha='center', va='top', fontsize=12, fontweight='normal', color='white')

# Impacto 3, Exposição 5: Síndrome Metabólica
ax.text(4 + 0.5, 2 + 0.65, "Síndrome\nMetabólica", ha='center', va='top', fontsize=12, fontweight='normal', color='black')

# Impacto 4, Exposição 5: Alzheimer
ax.text(4 + 0.5, 3 + 0.8, "Alzheimer", ha='center', va='top', fontsize=12, fontweight='normal', color='white')

# Adiciona nomenclaturas para Exposição igual a 1 (coluna 0)
# Impacto 1, Exposição 1: Intoxicação Exógena
ax.text(0 + 0.5, 0 + 0.65, "Intoxicação\nExógena", ha='center', va='top', fontsize=12, fontweight='normal', color='white')

# Impacto 3, Exposição 1: Síndrome Metabólica
ax.text(0 + 0.5, 2 + 0.65, "Síndrome\nMetabólica", ha='center', va='top', fontsize=12, fontweight='normal', color='white')

# Impacto 4, Exposição 1: Alzheimer
ax.text(0 + 0.5, 3 + 0.8, "Alzheimer", ha='center', va='top', fontsize=12, fontweight='normal', color='white')

# Adiciona textos "menos exposta" e "mais exposta"
ax.text(0 + 0.5, 0 - 0.02, "Menor\nexposição", ha='center', va='bottom', fontsize=12, color='black')
ax.text(4 + 0.5, 0 - 0.02, "Maior\nexposição", ha='center', va='bottom', fontsize=12, color='black')

plt.tight_layout()
plt.savefig('matriz_Risco_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### **PERCENTUAL DO RISCO**
# MAGIC
# MAGIC Percentual do Risco (Pr): é a quantificação do risco a partir adaptação da sua probabilidade de ocorrência para um índice percentual normalizado, utilizando-se o nível de risco (R) como base para essa conversão, de modo que o percentual pudesse ser usado como ponderador do valor financeiro do  (V). Para o cálculo, utilizou-se a normalização máxima e mínima para ajustar os valores do risco para o intervalo de 0% a 100%, como mostra a fórmula:
# MAGIC
# MAGIC ![image_1790130020926.png](./image_1790130020926.png "image_1790130020926.png")                                                    
# MAGIC
# MAGIC em que: R é o nível de risco; Rmin é o menor de nível de risco, igual a 1; Rmax é o maior de nível de risco, igual 25.
# MAGIC
# MAGIC O percentual do risco foi obtido para as duas distâncias entre a área de plantio e a comunidade, caracterizando os seguintes cenários espaciais:
# MAGIC - Menor distância: maior proximidade e maior exposição;
# MAGIC - Maior distância: menor proximidade e menor exposição.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Cálculo do Percentual do Risco
# ==========================================
# 1. ENTRADA DE DADOS
# ==========================================
# Valores dos resultados reais gerados na matriz determinística de pontuação, com escala de 1 a 5
impactos = {
    'Intoxicação Exógena': 1.0,     
    'Síndrome Metabólica': 3.0,     
    'Alzheimer': 4.0                
}

# Cenários de distância mapeados para o Nível de Exposição (E) correspondente
cenarios_distancia = {
    '87,7m (E: Critico)': {'distancia': 87.7, 'E': 5},
    '2255,56m (E: Muito Baixo)': {'distancia': 2255.56, 'E': 1}
}

# Limites operacionais de Risco (R = I * E)
# R_min = 1 * 1 = 1 | R_max = 5 * 5 = 25
R_min = 1.0
R_max = 25.0

# ==========================================
# 2. PROCESSAMENTO MATEMÁTICO DOS CENÁRIOS
# ==========================================
dados_calculados = []

for efeito, I in impactos.items():
    for label_cenario, info in cenarios_distancia.items():
        E = info['E']
        
        # Cálculo do Risco: R = I * E
        R = I * E
        
        # Cálculo da Percentual do Risco: Pr = ((R - Rmin) / (Rmax - Rmin)) * 100
        Pr = ((R - R_min) / (R_max - R_min)) * 100
        
        # Armazenando os resultados estruturados
        dados_calculados.append({
            'Efeito': efeito,
            'Cenário Espacial': label_cenario,
            'Impacto (I)': I,
            'Exposição (E)': E,
            'Risco (R)': R,
            'Percentual Risco (Pr %)': Pr
        })

# Convertendo para DataFrame para visualização
df_resultados = pd.DataFrame(dados_calculados)

print("=== RESULTADOS ===")
print(df_resultados.to_string(index=False))
print("\n" + "="*40 + "\n")


# COMMAND ----------

# MAGIC %md
# MAGIC #### **PERCENTUAL DO RISCO RESIDUAL**
# MAGIC
# MAGIC Percentual do Risco Residual (Pre): é a representação, em um índice percentual, do risco residual gerado após a implementação das ação de mitigação. É determinado pelo produto do percentual de risco (Pr) pela fração do coeficiente de eficácia (1 - e), que é o quanto a ação de controle reduz o risco.  O percentual de risco residual é uma adaptação do do conceito de risco residual (Vescio, 2022):
# MAGIC
# MAGIC ![image_1790171648514.png](./image_1790171648514.png "image_1790171648514.png")
# MAGIC
# MAGIC Coeficiente de eficácia (e): é o coeficiente das ações de controle planejadas para a mitigação do risco, sendo o escore que mede o quão eficazes são as ações para a redução do risco. Ele foi estimado por meio da lógica "fuzzy", criando-se uma matriz que correlaciona as ações de mitigação propostas conforme detalhado abaixo.
# MAGIC
# MAGIC **Ações de Mitigação do Risco**
# MAGIC
# MAGIC O plano de mitigação simplificado foi criado para ilustrar as ações de mitigação do risco riscos à saúde provocados pela deriva aérea. Ele foi concebido como um processo de melhoria contínua baseado no ciclo PDCA (Planejar, Executar, Verificar e Agir) do PMBOK (PMI, 2021), considerando as seguintes etapas:
# MAGIC - Planejamento para previsão das condições meteorológicas (Planejar);
# MAGIC - Execução da tecnologia de aplicação de pulverização (Executar);
# MAGIC - Monitoramento por medição meteorológica (Verificar e Agir).
# MAGIC
# MAGIC **Sistema de Inferência Fuzzy**
# MAGIC
# MAGIC Tipo de modelo:
# MAGIC - Optou-se pelo modelo Mamdami por permitir a criação de regras linguísticas, sendo possível alinhar com as ações de mitigação do risco baseado no ciclo PDCA, e por fornecer saída gradual e contínua, adequada à natureza difusa da eficácia.
# MAGIC
# MAGIC Variáveis linguísticas de entrada (ações de mitigação):
# MAGIC - Planejamento (meteorologia): Planejamento para previsão das condições meteorológicas;
# MAGIC - Tecnologia (pulverização): Execução da tecnologia de aplicação de pulverização;
# MAGIC - Monitoramento (meteorologia): Monitoramento por medição meteorológica.
# MAGIC
# MAGIC     Possui três antecedentes, cada um com universo discreto [0, 10]
# MAGIC
# MAGIC ![image_1790176976310.png](./image_1790176976310.png "image_1790176976310.png")
# MAGIC
# MAGIC Variáveis linguísticas de saída (coeficiente de eficácia):
# MAGIC
# MAGIC Coeficiente de eficácia (e) com cinco termos linguísticos, universo contínuo [0, 1] e passo 0,01
# MAGIC
# MAGIC ![image_1790177283422.png](./image_1790177283422.png "image_1790177283422.png")
# MAGIC
# MAGIC Base de Regras Fuzzy:
# MAGIC - Bloco 1 - Tecnologia de Aplicação Baixa (Regras C1–C9): sem a tecnologia adequada, as ações climáticas não conseguem conter a deriva, ainda que haja planejamento e monitoramento altos.
# MAGIC - Bloco 2 - Tecnologia de Aplicação Média (Regras C10–C18): a eficácia depende moderadamente do clima, apenas com planejamento e monitoramento alto se consegue alta eficácia, refletindo que a tecnologia média precisa de gestão robusta.
# MAGIC - Bloco 3 - Tecnologia de Aplicação Média (Regras C19–C27): garante-se alta eficácia se houver gestão do planejamento e monitoramento das ações climáticas.
# MAGIC
# MAGIC     Princípio básico das regras: a tecnologia define o teto de eficácia, e o planejamento e monitoramento determinam quão próximo do teto o sistema opera.
# MAGIC
# MAGIC Métodos de inferência e defuzzificação:
# MAGIC - Operador de implicação: mínimo;
# MAGIC - Agregação: máximo;
# MAGIC - Defuzzificação: centroide (centro de gravidade), padrão do scikit-fuzzy.
# MAGIC - Valor Crisp Representativo: Cada variável linguística de entrada é convertida para um valor crisp representativo, que são as aproximações arredondadas dos centroides das funções de pertinência de cada termo.
# MAGIC
# MAGIC ![image_1790180003707.png](./image_1790180003707.png "image_1790180003707.png")
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Cálculo do Coeficiente de Eficácia e Percentual do Risco Residual
# =============================================
# 0. BUSCAR VALORES DO DATAFRAME DA CÉLULA 13
# =============================================
# Salvar os riscos iniciais antes de df_resultados ser sobrescrito
df_riscos_celula13 = df_resultados.copy()
df_cenario_perto = df_riscos_celula13[df_riscos_celula13['Cenário Espacial'] == '87,7m (E: Critico)']
riscos_iniciais = {
    row['Efeito']: row['Percentual Risco (Pr %)']
    for _, row in df_cenario_perto.iterrows()
}

print("Riscos iniciais (Pr %) para o cenário 87,7m:")
for efeito, valor in riscos_iniciais.items():
    print(f"  {efeito}: {valor}")
print()

# ==========================================
# 1. DEFINIÇÃO DO UNIVERSO E VARIÁVEIS
# ==========================================
# Variáveis de Entrada (0 a 10)
pm = ctrl.Antecedent(np.arange(0, 11, 1), 'planejamento')
ta = ctrl.Antecedent(np.arange(0, 11, 1), 'tecnologia')
mm = ctrl.Antecedent(np.arange(0, 11, 1), 'monitoramento')

# Variável de Saída: Coeficiente de Eficácia e (0 a 1)
eficacia = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'eficacia')

# ==============================================
# 2. FUNÇÕES DE PERTINÊNCIA DE ENTRADA (0 A 10)
# ==============================================
for var in [pm, ta, mm]:
    var['baixa'] = fuzz.trimf(var.universe, [0, 0, 5])
    var['media'] = fuzz.trimf(var.universe, [2, 5, 8])
    var['alta'] = fuzz.trimf(var.universe, [5, 10, 10])

# ================================================
# 2. FUNÇÕES DE PERTINÊNCIA DA SAÍDA E (0 a 1)
# ================================================
# Ampliando de 3 para 5 níveis para maior sensibilidade
eficacia['muito_baixa'] = fuzz.trimf(eficacia.universe, [0, 0, 0.25])
eficacia['baixa']       = fuzz.trimf(eficacia.universe, [0.15, 0.3, 0.45])
eficacia['media']       = fuzz.trimf(eficacia.universe, [0.35, 0.5, 0.65])
eficacia['alta']        = fuzz.trimf(eficacia.universe, [0.55, 0.7, 0.85])
eficacia['muito_alta']  = fuzz.trimf(eficacia.universe, [0.75, 1.0, 1.0])

# ===========================================================
# 3. BASE DE REGRAS FUZZY (Cobrança do espaço de cenários)
# ===========================================================
regras = [
    # BLOCO 1: TECNOLOGIA BAIXA (Ações Climáticas não conseguem conter a deriva)
    ctrl.Rule(ta['baixa'] & pm['baixa'] & mm['baixa'], eficacia['muito_baixa']), # C1
    ctrl.Rule(ta['baixa'] & pm['baixa'] & mm['media'], eficacia['muito_baixa']), # C2
    ctrl.Rule(ta['baixa'] & pm['baixa'] & mm['alta'],  eficacia['muito_baixa']), # C3
    ctrl.Rule(ta['baixa'] & pm['media'] & mm['baixa'], eficacia['muito_baixa']), # C4
    ctrl.Rule(ta['baixa'] & pm['media'] & mm['media'], eficacia['baixa']), # C5
    ctrl.Rule(ta['baixa'] & pm['media'] & mm['alta'],  eficacia['baixa']), # C6
    ctrl.Rule(ta['baixa'] & pm['alta']  & mm['baixa'], eficacia['baixa']), # C7
    ctrl.Rule(ta['baixa'] & pm['alta']  & mm['media'], eficacia['media']), # C8
    ctrl.Rule(ta['baixa'] & pm['alta']  & mm['alta'],  eficacia['media']), # C9

    # BLOCO 2: TECNOLOGIA MÉDIA (Eficácia depende moderadamente do clima)
    ctrl.Rule(ta['media'] & pm['baixa'] & mm['baixa'], eficacia['muito_baixa']), # C10
    ctrl.Rule(ta['media'] & pm['baixa'] & mm['media'], eficacia['baixa']), # C11
    ctrl.Rule(ta['media'] & pm['baixa'] & mm['alta'],  eficacia['media']), # C12
    ctrl.Rule(ta['media'] & pm['media'] & mm['baixa'], eficacia['media']), # C13
    ctrl.Rule(ta['media'] & pm['media'] & mm['media'], eficacia['media']), # C14
    ctrl.Rule(ta['media'] & pm['media'] & mm['alta'],  eficacia['media']), # C15
    ctrl.Rule(ta['media'] & pm['alta']  & mm['baixa'], eficacia['media']), # C16
    ctrl.Rule(ta['media'] & pm['alta']  & mm['media'], eficacia['media']), # C17
    ctrl.Rule(ta['media'] & pm['alta']  & mm['alta'],  eficacia['alta']), # C18

    # BLOCO 3: TECNOLOGIA ALTA (Garante alta eficácia se houver gestão)
    ctrl.Rule(ta['alta'] & pm['baixa'] & mm['baixa'], eficacia['media']), # C19
    ctrl.Rule(ta['alta'] & pm['baixa'] & mm['media'], eficacia['media']), # C20
    ctrl.Rule(ta['alta'] & pm['baixa'] & mm['alta'],  eficacia['alta']), # C21
    ctrl.Rule(ta['alta'] & pm['media'] & mm['baixa'], eficacia['media']), # C22
    ctrl.Rule(ta['alta'] & pm['media'] & mm['media'], eficacia['alta']),  # C23
    ctrl.Rule(ta['alta'] & pm['media'] & mm['alta'],  eficacia['muito_alta']),  # C24
    ctrl.Rule(ta['alta'] & pm['alta']  & mm['baixa'], eficacia['alta']),  # C25
    ctrl.Rule(ta['alta'] & pm['alta']  & mm['media'], eficacia['muito_alta']),  # C26
    ctrl.Rule(ta['alta'] & pm['alta']  & mm['alta'],  eficacia['muito_alta'])   # C27
]

sistema_controle = ctrl.ControlSystem(regras)
simulacao = ctrl.ControlSystemSimulation(sistema_controle)

# ==========================================================================
# 4. GERAÇÃO DOS 27 CENÁRIOS
# ==========================================================================
# Nota: riscos_iniciais já foi definido no início da célula
# Valores representativos (crisp) para cada termo linguístico
niveis = {'Baixa': 1.0, 'Média': 5.0, 'Alta': 9.0}

resultados = []
cenario_id = 1

# Loop iterando pelas 3 ações e seus 3 níveis (3x3x3 = 27 cenários)
for nome_ta, val_ta in niveis.items():
    for nome_pm, val_pm in niveis.items():
        for nome_mm, val_mm in niveis.items():
            
            # Inserindo dados na simulação
            simulacao.input['tecnologia'] = val_ta
            simulacao.input['planejamento'] = val_pm
            simulacao.input['monitoramento'] = val_mm
            
            # Computando o resultado
            try:
                simulacao.compute()
                e = simulacao.output['eficacia']
            except:
                e = 0.0 # Fallback caso alguma regra não cubra uma transição extrema
            
            # Cálculo do Risco Residual (Pre = Pr * (1 - e))
            pre_alzheimer = riscos_iniciais['Alzheimer'] * (1 - e)
            pre_sindrome = riscos_iniciais['Síndrome Metabólica'] * (1 - e)
            pre_intox = riscos_iniciais['Intoxicação Exógena'] * (1 - e)
            
            # Salvando os dados
            resultados.append({
                'Cenário': f"C{cenario_id}",
                'Tecnologia (Do)': nome_ta,
                'Planejamento (Plan)': nome_pm,
                'Monitoramento (Check)': nome_mm,
                'Plan_Check': f"{nome_pm} / {nome_mm}", # Eixo Y do Heatmap
                'Eficácia (e)': e,
                'Pre_Alzheimer (%)': pre_alzheimer,
                'Pre_Sindrome (%)': pre_sindrome,
                'Pre_Intox (%)': pre_intox
            })
            cenario_id += 1

# Criando DataFrame para análise e exportação
df_resultados = pd.DataFrame(resultados)
print(df_resultados.to_string(index=False))



# COMMAND ----------

# DBTITLE 1,Figura da Matriz do Coeficiente de Eficácia
# ==========================================
# 5. PLOTAGEM DO MAPA DE CALOR
# ==========================================
# Pivotando a tabela para o formato de matriz (9 linhas x 3 colunas)
heatmap_data = df_resultados.pivot(index='Plan_Check', columns='Tecnologia (Do)', values='Eficácia (e)')

# Reordenando as linhas e colunas para fazer sentido lógico (Baixo -> Alto)
ordem_linhas = ['Baixa / Baixa', 'Baixa / Média', 'Baixa / Alta', 
                'Média / Baixa', 'Média / Média', 'Média / Alta', 
                'Alta / Baixa', 'Alta / Média', 'Alta / Alta']
ordem_colunas = ['Baixa', 'Média', 'Alta']

heatmap_data = heatmap_data.reindex(index=ordem_linhas, columns=ordem_colunas)

# Configurando o gráfico
plt.figure(figsize=(10, 8))
ax = sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlGn", 
                 linewidths=0, cbar_kws={'label': 'Coeficiente de Eficácia'},
                 vmin=0, vmax=1, annot_kws={"fontsize":13})

# Ajusta fonte e negrito da label do colorbar
cbar = ax.collections[0].colorbar
cbar.set_label('Coeficiente de Eficácia', fontsize=13, fontweight='bold')
cbar.ax.tick_params(labelsize=11)  # Aumenta tamanho dos números do eixo do coeficiente de eficácia

# Inserir a palavra "Ações" abaixo do número do quadrante alto / alto x médio
# Linha: 'Alto / Alto' (índice 8), Coluna: 'Médio' (índice 1)
ax.text(1 + 0.5, 8 + 0.70, "Ações de Mitigação", ha='center', va='top', fontsize=13, fontweight='normal', color='black')

#plt.title('Mapa de Calor dos 27 Cenários: Coeficiente de Eficácia (E)', fontsize=14, pad=15)
plt.ylabel('METEOROLOGIA\n(Planejamento / Monitoramento)', fontsize=15, fontweight='bold')
plt.xlabel('PULVERIZAÇÃO\n(Aplicação)', fontsize=15, fontweight='bold')

# Ajusta fonte dos textos dos eixos para 11
ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=11)

# Ajustando layout e salvando a imagem em alta resolução (Ideal para o TCC)
plt.tight_layout()
plt.savefig('matriz_eficacia.png', dpi=300)
plt.show()