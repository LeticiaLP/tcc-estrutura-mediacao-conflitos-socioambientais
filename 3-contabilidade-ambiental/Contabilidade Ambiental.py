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
# MAGIC - Módulo III: Planejamento da Mitigação do Risco
# MAGIC - Módulo IV: Contabilidade Ambiental
# MAGIC
# MAGIC ------------------------------

# COMMAND ----------

# DBTITLE 1,Imports
from pyspark.sql.functions import col
from pyspark.sql.functions import avg
from pyspark.sql import functions as F
from pyspark.sql.types import *

import requests
import json

import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC #### **IMPACTOS FINANCEIROS**
# MAGIC
# MAGIC Os impactos financeiros são estimados pelos valores monetários relacionados ao risco socioambiental identificado pelo uso de agrotóxicos nas plantações, sendo os valores que podem ser gastos quando não se tem a ação para mitigar os riscos.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ##### **Impacto Financeiro por Perda de Produtividade (IFpp)**
# MAGIC
# MAGIC O cálculo do **impacto financeiro por perda de produtividade (IFpp)** por afastamento do trabalho ou invalidez  está na fórmula abaixo, e tem como base conceitual para sua criação o método de custeio de absenteísmo e rotatividade proposto por Cascio e Boudreau (2010) e a fórmula do valor presente sob a ótica do judiciário brasileiro (Pinto Júnior, 2024) para invalidez permanente.
# MAGIC
# MAGIC ![image_1790190789688.png](./image_1790190789688.png "image_1790190789688.png")
# MAGIC
# MAGIC em que: AFt é o custo do afastamento do trabalho temporariamente; Ip é a indenização por invalidez permanente; Cr é o custo de rotatividade.
# MAGIC
# MAGIC **1 Premissas Adotadas**
# MAGIC
# MAGIC - Para afastamento temporário do trabalho considerou-se 15 dias, pois é o prazo máximo pago pelo empregador segundo o artigo 60 da Lei nº 8.213/91 (https://www.planalto.gov.br/ccivil_03/leis/l8213cons.htm).
# MAGIC - O rendimento mensal médio do cargo (Rm) foi calculado como a média do salário no ano de 2025, para o CBO 622730 (trabalhador na cultura de soja) no Estado do Pará, extraído da base de dados pública do Cadastro Geral de Empregados e Desempregados (CAGED).
# MAGIC - Para o cáculo do valor médio da diária de trabalho (Vd), usou-se o divisor fixo de 30 dias para obtenção da diária dos funcionários mensalistas, independentemente de o mês ter 28, 29 ou 31 dias, segundo o artigo 64 do Decreto-lei nº 5.452 (https://www.planalto.gov.br/ccivil_03/decreto-lei/del5452.htm).
# MAGIC - Foi adotada para o estudo o percentual de invalidez da doença na tabela de Invalidez Permanente por Acidente (IPA) da SUSEP (Superintendência de Seguros Privados) como uma variável substituta (proxy) para gravidade clínica.
# MAGIC - Para a taxa SELIC (r), fez-se a média aritmética simples da taxa básica de juros diária anualizada (% a.a.) para o ano-base de 2025. Usada para manter a consistência temporal com a base salarial de referência (Rm) utilizada na composição dos custos de produtividade.
# MAGIC - O tempo médio do ciclo de vida da doença incapacitante em anos (t) foi obtido segundo os Protocolos Clínicos e Diretrizes Terapêuticas (PCDT) do Ministério da Saúde (MS), no qual tem-se um período médio entre o diagnóstico e a morte pela doença.
# MAGIC - Para o cálculo do valor financeiro por perda de produtividade com (Vpp) e sem mitigação (Vppm) do risco, adotou-se os índices de percentual do risco e do risco residual da comunidade próxima à plantyação como “proxy” para o percentual do risco do trabalhador, a partir da premissa simplificada de que o uso de equipamento de proteção individual poderia atuar como uma barreira, reduzindo a exposição e equiparando os riscos.
# MAGIC
# MAGIC
# MAGIC **2 Custos por afastamento do trabalho (AFt)**
# MAGIC
# MAGIC Os **custos por afastamento do trabalho (AFt)** temporariamente representam o valor pago pelo empregador durante a interrupção da atividade do trabalhador devido a uma doença, que corresponde ao próprio salário, e estão na form. (6), sendo a adaptação do custo de absenteísmo de Cascio e Boudreau (2010):
# MAGIC
# MAGIC ![image_1784067004663.png](./image_1784067004663.png "image_1784067004663.png")                                                 
# MAGIC
# MAGIC em que: 15 é o número máximo de dias de afastamento do trabalho pagos pelo empregador; Vd é o valor médio da diária de trabalho para CBO 622730 (trabalhador na cultura de soja), com divisor fixo de 30 dias para obtenção da diária dos funcionários mensalistas.
# MAGIC
# MAGIC Para calcular o valor médio da diária de trabalho (Vd), será feita uma adaptação da abordagem legal do padrão CLT (Consolidação das Leis do Trabalho), usando-se o divisor fixo de 30 dias para obtenção da diária dos funcionários mensalistas, independentemente de o mês ter 28, 29 ou 31 dias (Brasil, 1943). Desta forma, o cálculo do valor médio da diária de trabalho (Vd) será a divisão entre rendimento mensal médio do cargo (Rm) e os 30 dias referentes ao mês de trabalho.
# MAGIC
# MAGIC ![image_1784043724933.png](./image_1784043724933.png "image_1784043724933.png")
# MAGIC
# MAGIC **3 Indenização em caso de invalidez permanente (Ip)**
# MAGIC
# MAGIC A **indenização em caso de invalidez permanente (Ip)** corresponde a exposição recorrente do empregado ao contaminante, que gera uma doença incapacitante, e acarreta custos futuros referentes a uma pensão mensal, pleiteada judicialmente, sendo calculada tendo como referência a fórmula do valor presente sob a ótica do judiciário brasileiro (Pinto Júnior, 2024):
# MAGIC
# MAGIC ![image_1784042012071.png](./image_1784042012071.png "image_1784042012071.png")                                                    
# MAGIC
# MAGIC em que: Rm é o rendimento mensal médio do cargo, multiplicado pelos meses de salário no ano, incluindo o 13° e o terço constitucional de férias; Pi é o percentual de invalidez para cada tipo de doença; (1+r)^t é o fator de desconto, que considera o percentual da taxa SELIC (r), ao longo do tempo médio do ciclo de vida de uma doença incapacitante (t) em anos. 
# MAGIC
# MAGIC Como limitação metodológica do modelo de valoração do Impacto por Invalidez Permanente (Ip), destaca-se o uso da taxa Selic histórica de 2025 de forma linear ao longo de todo o horizonte temporal de projeção (t). Por se tratar de uma taxa de desconto nominal e estática aplicada a rendimentos constantes (Rm), o modelo assume uma simplificação que gera uma subvaloração do impacto financeiro nos períodos mais distantes devido ao efeito da inflação implícita no denominador da equação. No entanto, essa escolha metodológica justifica-se pela necessidade de manter a consistência temporal dos dados com o ano-base dos salários de referência de 2025, servindo como uma estimativa conservadora do passivo socioambiental estudado.
# MAGIC
# MAGIC **4 custos de rotatividade (Cr)**
# MAGIC
# MAGIC Os **custos de rotatividade (Cr)** mensura quanto se gasta para substituir um funcionário devido a sua saída em decorrência da invalidez. De acordo com Cascio e Boudreau (2010), incluem custos de desligamento, reposição e treinamento, além das diferenças de desempenho entre quem sai e quem entra, e podem chegar a 150% ou mais do salário anual do funcionário que está saindo. Considerando a abordagem metodológica do trabalho, será usada uma simplificação da teoria de Cascio e Boudreau para o cálculo do custo de rotatividade, mostrada na form. (9) como sendo o produto entre o rendimento mensal médio do cargo (Rm), os meses de salário no ano, incluindo o 13° e o terço constitucional de férias, e o fator do custo de 150%.
# MAGIC
# MAGIC ![image_1784042023070.png](./image_1784042023070.png "image_1784042023070.png")                                              (9)
# MAGIC
# MAGIC em que: Rm é o rendimento mensal médio do cargo.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Função de Impacto Financeiro por Perda de Produtividade
def calcular_aft(da: int, vd: float) -> float:
    """
    Custo do afastamento do trabalho temporariamente (AFt)
    da: Número de dias de afastamento do trabalho.
    vd: Valor médio da diária de trabalho.
    """

    aft = da * vd    

    return aft

def calcular_ip(rm: float, pi: float, r: float, t_anos: int) -> float:
    """
    Indenização em caso de invalidez permanente (Ip)
    rm: Rendimento mensal médio do cargo.
    pi: Percentual de invalidez.
    r: Taxa SELIC (decimal).
    t_anos: Tempo médio do ciclo de vida da doença incapacitante em anos.
    """
    
    ip = 0.0
    for t in range(1, t_anos + 1):
        ip += ((rm * 13.3) * pi) / ((1 + r) ** t)

    return ip

def calcular_cr(rm: float) -> float:
    """
    Custos de rotatividade (Cr)
    rm: Rendimento mensal médio do cargo.
    """

    cr = (rm * 13.3) * 1.5

    return cr

def calcular_ifpp(aft: float, ip: float, cr: float) -> float:
    """
    Impacto Financeiro por Perda de Produtividade (IFpp)
    """

    ifpp = aft + ip + cr

    return ifpp

# COMMAND ----------

# DBTITLE 1,Cálculo do Impacto Financeiro por Perda de Produtividade
"""
PARA INTOXICAÇÃO EXÓGENA E SÍNDROME METABÓLICA
Custo por afastamento do trabalho (AFt) temporariamente: produto dos dias de afastamento do trabalho pelo valor médio da diária de trabalho.
Dias de afastamento do trabalho máximo pagos pelo empregador é de 15 dias.
Valor médio da diária de trabalho (Vd) de 2025 para CBO 622730 (trabalhador na cultura de soja) no Pará extraído da base de dados filtrada do CAGED no google cloud e importada via .json para o Databricks
"""
# Rendimento médio mensal do trabalhador na cultura de soja em 2025 no Pará
'''
Cadastro Geral de Empregados e Desempregados (CAGED): Tabela tratada Microdados de Movimentações
Fonte: https://basedosdados.org/dataset/562b56a3-0b01-4735-a049-eeac5681f056?table=2245875f-d1ef-490d-be29-4f8fb2191335
'''
df_rendimento = (
    spark.table("tcc_projetos.caged_brz.caged")
    .select(col("salario_mensal").cast("float"))
)

rendimento_medio = df_rendimento.agg(avg("salario_mensal").alias("media")).first()["media"]

# Valor médio da diária de trabalho (Vd)
valor_diaria = rendimento_medio / 30

# Custos por afastamento do trabalho (AFt) temporariamente para 15 dias de afastamento
dias_afastamento = 15

# Custo por afastamento do trabalho (AFt) temporariamente 
AFt = calcular_aft(dias_afastamento, valor_diaria)

print(f"Rendimento médio do salário (Rm): {rendimento_medio}")
print(f"Valor da diária de trabalho (Vd): {valor_diaria}")
print(f"Custo por afastamento do trabalho (AFt) por 15 dias: {AFt:.2f}")

"""
PARA ALZHEIMER
Custo de indenização em caso de invalidez permanente (Ip): produto do rendimento médio anual do trabalhador pelo percentual de invalidez da doença incapacitante divido pelo fator de desconto, que é o pecentual da taxa selic ao longo da vida da doença incapacitante.
Rendimento médio (Rm) do trabalhador na cultura de soja em 2025 no Pará.
Percentual de invalidez (Pi) para cada tipo de doença. Será adotada para o estudo a tabela de Invalidez Permanente por Acidente (IPA) da SUSEP (Superintendência de Seguros Privados) como uma variável substituta (proxy) de gravidade clínica.
Taxa SELIC (r) é média aritmética simples da taxa básica de juros diária anualizada (% a.a.) para o ano-base de 2025. Usada para manter a consistência temporal com a base salarial de referência (Rm) utilizada na composição dos custos de produtividade.
Tempo médio do ciclo de vida da doença incapacitante em anos (t).
"""
# Cálculo da média aritmética da taxa de juros (taxa SELIC) diária anualizada (% a.a.) para 2025
'''
Banco Central do Brasil (BCB): Dados diários
Filtros aplicados: Data inicial: 02/01/2025 / Data final: 31/12/2025
Fonte: https://www.bcb.gov.br/estabilidadefinanceira/selicdadosdiarios
'''
df_selic = (
    spark.table("tcc_projetos.bc_brz.selic")
    .select(col("taxa_aa").cast("float"))
)
selic_media = df_selic.agg(avg("taxa_aa").alias("media")).first()["media"]

taxa_selic = selic_media / 100

print(f"Taxa SELIC média para 2025: {taxa_selic}")

# Para Invalidez por Alzheimer em estágio avançado, com danos progressivos e irreversíveis.
# PERCENTUAL DE INVALIDEZ (Pi): O estágio avançado do Alzheimer tem equivalência clínica com a classificação da SUSEP (2010) para "Lesões neurológicas que cursem com dano cognitivo-comportamental alienante, impedimento do senso de orientação espacial e perda do controle esfincteriano ou de funções vitais', caracterizando a perda total da capacidade civil e laborativa do indivíduo". Por isso, adotou-se a perda funcional máxima para percentual de invalidez (Pi = 1, equivalente a 100%).
percentual_invalidez = 1        # 100%

# TEMPO MÉDIO DO CICLO DE VIDA DA DOENÇA INCAPACITANTE (t): Segundo os Protocolos Clínicos e Diretrizes Terapêuticas (PCDT) do Ministério da Saúde, o período médio entre o diagnóstico e a morte na doença de Alzheimer varia, em geral, entre três e oito anos, sendo que essa estimativa depende diretamente da idade do indivíduo no momento do diagnóstico. Para o estudo, será adotado o tempo máximo de 8 anos.
tempo_anos = 8                  # tempo médio máximo

# Custo da indenização em caso de invalidez permanente (Ip)
Ip = calcular_ip(rendimento_medio, percentual_invalidez, taxa_selic, tempo_anos)

print(f"Custo de indenização em caso de invalidez permanente (Ip): {Ip:.2f}")

"""
Custo de rotatividade (Cr):  150% do salário anual (salário + 1/3 de férias) do funcionário que está saindo
Rendimento médio do trabalhador na cultura de soja em 2025 no Pará
"""
Cr = calcular_cr(rendimento_medio)

print(f"Custo de rotatividade (Cr): {Cr:.2f}")

"""
Impacto Financeiro por Perda de Produtividade (IFpp)
Considerando um afastamento do trabalho temporário no ano para duas doenças (intoxicação exógena e síndrome metabólica), uma indenização em caso de invalidez permanente para uma doença (Alzheimer) e um custo de rotatividade no ano.
"""
# Cálculo para síndrome metabólica e intoxicação exógena
ifpp_sm_ie = calcular_ifpp(AFt, 0, 0)

# Cálculo para síndrome metabólica e intoxicação exógena
ifpp_a = calcular_ifpp(0, Ip, Cr)

print(f"Impacto Financeiro por Perda de Produtividade (IFpp_ie): {ifpp_sm_ie:.2f}")
print(f"Impacto Financeiro por Perda de Produtividade (IFpp_sm): {ifpp_sm_ie:.2f}")
print(f"Impacto Financeiro por Perda de Produtividade (ifpp_a): {ifpp_a:.2f}")


# COMMAND ----------

# MAGIC %md
# MAGIC ##### **Impacto Financeiro por Processos Judiciais (IFpj)**
# MAGIC
# MAGIC O cálculo do **impacto financeiro por processos judiciais (IFpj)** abertos na região do conflito está na fórmula abaixo, e é baseado nos conceitos de Jurimetria e nas normas de contabilidade internacional para gerar uma estimava de valor (Silva e Coutinho, 2019):
# MAGIC
# MAGIC ![image_1790254700799.png](./image_1790254700799.png "image_1790254700799.png")                                                
# MAGIC
# MAGIC em que: Vi é o valor médio de indenizações pagas em processos da região.
# MAGIC
# MAGIC **1 Premissas Adotadas**
# MAGIC
# MAGIC - Na ausência de dados estruturados sobre os valores das indenizações pagas nos processos judiciais, foram usados dados agregados das Varas do Trabalho do TRT da 8º Região (PA-AP) como proxy neste estudo para simular os valores nos diversos Tribunais.
# MAGIC - Usou-se um arquivo excel da Justiça do Trabalho relacionado a casos recebidos e julgados nas Varas do Trabalho (https://www.tst.jus.br/web/estatistica/vt/recebidos-e-julgados#).
# MAGIC - A base não possui distinção da atividade econômica e assunto dos valores pagos, por isso fez-se o cálculo para todos os processos do TRT da 8º Região (PA-AP).
# MAGIC - Fez-se a média do valor pago de 3 anos (2023 a 2025).
# MAGIC - Para o cálculo do valor financeiro do risco por processo judicial com e sem mitigação do risco, os percentuais do risco e do risco residual têm relação direta com a comunidade próxima, constituindo um fato para a possível materialização de passivos cíveis, coletivos e ambientais.
# MAGIC
# MAGIC **2 Valor médio de indenizações pagas (Vi)**
# MAGIC
# MAGIC Para obter o valor das indenizações, somou-se todos os valores pagos das Varas do TRT 8º Região por ano e dividiu-se pela soma de todos os processos julgados (Conciliações e Julgados - excluídos acordos, desistências e arquivamentos) por ano, do qual extraiu-se uma média dos anos tratados.
# MAGIC
# MAGIC ![image_1790254677102.png](./image_1790254677102.png "image_1790254677102.png")
# MAGIC

# COMMAND ----------

# DBTITLE 1,Cálculo do Impacto Financeiro por Processos Judiciais
# Selecionar campos e filtrar a tabela para os anos de 2023 a 2025
'''
Vara de Trabalho: Tabela Base de Dados completa JT, em excel
Fonte: https://www.tst.jus.br/web/estatistica/vt/recebidos-e-julgados
'''

df_vara_trabalho = (
    spark.table("tcc_projetos.trt_brz.vara_trabalho")
    .select(col("Variavel"),
            col("Atividade_Economica"),
            col("Assunto"),
            col("Solucao"),
            col("Ano"),
            col("Quantidade"))
    .filter(col("Ano").between(2023, 2025))
    .filter(col("Regiao") == 8)
    .filter(col("Variavel").isin(["Julgados", "Valores Pagos", "CN por Assunto", "Casos Novos"]))
)

'''
Vi valor médio de indenizações pagas em processos da região
'''
# Cálculo da soma do valores pagos em indenizações por ano (2023 a 2025)
df_valores_pagos = (
    df_vara_trabalho
    .filter(col("Variavel") == "Valores Pagos")
    .groupBy("Ano")
    .agg(F.sum("Quantidade").alias("soma_vp"))
)

# Cálculo da soma dos números de processo julgados (Conciliações, Julgados - excluídos acordos, desistências e arquivamentos, Julgados excluídos acordos, desistências e arquivamentos) por ano (2023 a 2025)
df_processo_julgado = (
    df_vara_trabalho
    .filter(col("Variavel")== "Julgados")
    .filter((col("Solucao") == "Conciliações") | col("Solucao").like("%Julgado%"))
    .groupBy("Ano")
    .agg(F.sum("Quantidade").alias("soma_pj").cast("int"))
)

# Cálculo do valor médio das indenizações (Vi) por ano (2023 a 2025)
df_vi_anual = (
    df_valores_pagos.join(df_processo_julgado, on="Ano")
    .select(
        col("Ano"),
        col("soma_vp").alias("valores_pagos_soma"),
        col("soma_pj").alias("processo_julgado_soma"),
        (col("soma_vp") / col("soma_pj")).alias("valor_medio_indenizacao")
    )
)

# Cálculo da média do valor médio das indenizações (Vi) por ano (2023 a 2025)
df_valor_indenizacao = (
    df_vi_anual
    .select("valor_medio_indenizacao")
    .agg(F.avg("valor_medio_indenizacao").alias("media_vi"))
)

# Extrair valores escalares dos DataFrames
valor_indenizacao = df_valor_indenizacao.first()["media_vi"]

# Cálculo do Impacto Financeiro por Processos Judiciais
ifpj = valor_indenizacao


print(f"Valor médio de indenização (Vi): R$ {valor_indenizacao:.2f}")
print(f"Impacto Financeiro por Processos Judiciais (ifpj): R$ {ifpj:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### **VALOR FINANCEIRO PONDERADO DO RISCO (V)**
# MAGIC
# MAGIC O valor financeiro ponderado do risco foi obtido pela adaptação simplificada da fórmula do Valor Monetário Esperado (VME), que é uma técnica para quantificaçao das incertezas (Project Management Institute [PMI], 2021).
# MAGIC
# MAGIC O risco socioambiental é quantificado monetariamente pelo valor financeiro, a partir do produto do impacto financeiro pelo índice percentual normalizado do risco. 
# MAGIC
# MAGIC Para o estudo, foram obtidos os valores financeiros ponderados sem e com mitigação dos riscos, denominados valor financeiro do risco sem prática sustentável (Vr) e valor financeiro do risco com prática sustentável (Vm).
# MAGIC
# MAGIC **1 Premissas Adotadas**
# MAGIC
# MAGIC - Para o cálculo do valor financeiro do risco por perda de produtividade com (Vppm) e sem mitigação do risco (Vpp), adotou-se o percentual do risco (Pr) e do risco residual (Pre) da comunidade próxima como “proxy” para o risco do trabalhador, a partir da premissa de que o uso de equipamento de proteção individual atua como uma barreira, reduzindo a exposição. 
# MAGIC - Para o cálculo do valor financeiro do risco por processo judicial com (Vpjm) e sem mitigação do risco (Vpj), os percentuais do risco e do risco residual têm relação direta com a comunidade próxima, constituindo um fato para a possível materialização de passivos cíveis, coletivos e ambientais.
# MAGIC - Foi considerado para o cáculo do valor financeiro apenas uma ocorrência de cada impacto financeiro por agravo de saúde em um ano, equivalente a:
# MAGIC     - uma perda de produtividade e um processo judicial no ano para intoxicação exógena;
# MAGIC     - uma perda de produtividade e um processo judicial no ano para síndrome metabólica;
# MAGIC     - uma perda de produtividade e um processo judicial no ano para Alzheimer;
# MAGIC
# MAGIC **2 Valor Financeiro do Risco sem prática sustentável (Vr)**
# MAGIC
# MAGIC O valor financeiro do risco (Vr) é calculado pela fórmula abaixo:
# MAGIC
# MAGIC ![image_1790260842715.png](./image_1790260842715.png "image_1790260842715.png")
# MAGIC
# MAGIC em que: Vpp é o valor financeiro do risco por perda de produtividade; Vpj é valor financeiro do risco por processo judicial.
# MAGIC
# MAGIC O valor financeiro do risco por perda de produtividade (Vpp) se encontra na fórmula:
# MAGIC
# MAGIC ![image_1790260973768.png](./image_1790260973768.png "image_1790260973768.png")
# MAGIC
# MAGIC em que: Pr é o percentual do risco de cada agravo de saúde; IFpp é o seu impacto financeiro por perda de produtividade de cada agravo de saúde.
# MAGIC
# MAGIC O valor financeiro do risco por processo judicial [Vpj]  é apresentado na fórmula:
# MAGIC
# MAGIC ![image_1790261011605.png](./image_1790261011605.png "image_1790261011605.png")
# MAGIC
# MAGIC em que: Pr é o percentual de risco de cada agravo de saúde; IFpj é o seu impacto financeiro por processo judicial.
# MAGIC
# MAGIC **3 Valor Financeiro Ponderado do risco após a implementação da sua mitigação (Vm)**
# MAGIC
# MAGIC O valor financeiro ponderado do risco após a implementação da sua mitigação (Vm) está na fórmula abaixo, sendo uma adaptação para a quantificação do dano residual ao substituir a probabilidade por um índice percentual (PMI, 2021)::
# MAGIC
# MAGIC ![image_1790261213019.png](./image_1790261213019.png "image_1790261213019.png")
# MAGIC
# MAGIC em que: Vppm é o valor financeiro do risco por perda de produtividade após a mitigação; Vpjm é valor financeiro do risco por processo judicial após a mitigação.
# MAGIC
# MAGIC O vvalor financeiro do risco por perda de produtividade após a implantação das ações de mitigação (Vppm) está na fórmula:
# MAGIC
# MAGIC ![image_1790261272106.png](./image_1790261272106.png "image_1790261272106.png")
# MAGIC
# MAGIC em que: Pre é o percentual do risco residual de cada agravo de saúde após implementação da mitigação; IFpp é o impacto financeiro por perda de produtividade de cada agravo de saúde.
# MAGIC
# MAGIC O vvalor financeiro do risco por processo judicial após a implementação das ações de mitigação (Vpjm) está na fórmula:
# MAGIC
# MAGIC ![image_1790261310522.png](./image_1790261310522.png "image_1790261310522.png")
# MAGIC
# MAGIC em que: Pre é o percentual do risco residual de cada agravo de saúde após implementação da mitigação; IFpj é o impacto financeiro por processo judicial.

# COMMAND ----------

# DBTITLE 1,Função do Valor Financeiro Ponderado do Risco
def calcular_vr(vpp: float, vpj: float) -> float:
    """
    Valor financeiro ponderado do risco (Vr)
    vpp: valor financeiro do risco por perda de produtividade (Vpp)
    vpj: valor financeiro do risco por processo judicial (Vpj)
    """
    vr = vpp + vpj

    return vr


def calcular_vm(vppm: float, vpjm: float) -> float:
    """
    Valor financeiro do risco após a implementação da sua mitigação (Vm)
    vppm: valor financeiro do risco por perda de produtividade após a mitigação (Vppm)
    vpjm: valor financeiro do risco por processo judicial após a mitigação (Vpjm)
    """
    vm = vppm + vpjm

    return vm

def calcular_vf(pr1: float, if1: float, pr2: float, if2: float, pr3: float, if3: float) -> float:
    """
    Calcular valor financeiro do risco por perda de produtividade com e sem mitigação (Vpp e Vppm)
    Calcular valor financeiro do risco por processo judicial com e sem mitigação (Vpj e Vpjm)
    """
   
   # Criar listas para iterar
    percentuais_risco = [pr1, pr2, pr3]
    impactos_financeiros = [if1, if2, if3]
    agravos = ["Intoxicação Exógena", "Síndrome Metabólica", "Alzheimer"]
    
    # Loop para calcular Vpp = Σ(Pr_i/100 × IFpp_i)
    vf = 0.0
    print("\nDetalhamento por agravo de saúde:\n")

    for i in range(len(agravos)):
        parcela = (percentuais_risco[i] / 100) * impactos_financeiros[i]
        vf += parcela
        print(f"{agravos[i]:25} | Pr = {percentuais_risco[i]:6.2f}% | IFpp = R$ {impactos_financeiros[i]:12,.2f} | Parcial = R$ {parcela:12,.2f}")

    return vf

# COMMAND ----------

# DBTITLE 1,Cálculo do Valor Financeiro do Risco
# Percentual de Risco (Pr) para cada agravo - valores em %
pr1 = 16.666667   # intoxicação exógena
pr2 = 58.333333    # síndrome metabólica
pr3 = 79.166667    # Alzheimer

"""
Valor financeiro do risco por perda de produtividade (Vpp)
Vpp = Σ(Pr_i/100 × IFpp_i)
"""
# IFpp correspondente para cada agravo
ifpp1 = ifpp_sm_ie  # intoxicação exógena
ifpp2 = ifpp_sm_ie  # síndrome metabólica
ifpp3 = ifpp_a      # Alzheimer

# Cálculo de Vpp
print("="*80)
print(" CÁLCULO DO VALOR FINANCEIRO DO RISCO POR PERDA DE PRODUTIVIDADE (Vpp)")
print("="*80)

vpp = calcular_vf(pr1, ifpp1, pr2, ifpp2, pr3, ifpp3)

print("-"*80)
print(f"\nVALOR FINANCEIRO DO RISCO POR PERDA DE PRODUTIVIDADE (Vpp): R$ {vpp:,.2f}")
print("="*80)

"""
Valor financeiro do risco por processo judicial (Vpj)
Vpj = Σ(Pr_i/100 × IFpj)
"""
# IFpj correspondente para cada agravo
ifpj1 = ifpj  # intoxicação exógena
ifpj2 = ifpj  # síndrome metabólica
ifpj3 = ifpj  # Alzheimer

# Cálculo de Vpp
print("="*80)
print(" CÁLCULO DO VALOR FINANCEIRO DO RISCO POR PROCERSSO JUDICIAL (Vpj)")
print("="*80)

vpj = calcular_vf(pr1, ifpj1, pr2, ifpj2, pr3, ifpj3)

print("-"*80)
print(f"\nVALOR FINANCEIRO DO RISCO POR PROCERSSO JUDICIAL (Vpj): R$ {vpj:,.2f}")
print("="*80)

# Cálculo do Valor financeiro do Risco (Vr)
vr = calcular_vr(vpp, vpj)

print("="*80)
print(" CÁLCULO DO VALOR FINANCEIRO DO RISCO (Vr)")
print("="*80)

print("-"*80)
print(f"\nVALOR FINANCEIRO DO RISCO (Vr): R$ {vr:,.2f}")
print("="*80)


# COMMAND ----------

# DBTITLE 1,Cálculo do Valor Financeiro do Risco após Mitigação
# Percentual de Risco Residual após Mitigação (Pre) - valores em %
pre1 = 5.00     # intoxicação exógena
pre2 = 17.5     # síndrome metabólica
pre3 = 23.75    # Alzheimer

"""
Valor financeiro do risco por perda de produtividade (Vpp)
Vpp = Σ(Pr_i/100 × IFpp_i)
"""
# IFpp correspondente para cada agravo
ifpp1 = ifpp_sm_ie  # intoxicação exógena
ifpp2 = ifpp_sm_ie  # síndrome metabólica
ifpp3 = ifpp_a      # Alzheimer

# Cálculo de Vpp
print("="*80)
print(" CÁLCULO DO VALOR FINANCEIRO DO RISCO POR PERDA DE PRODUTIVIDADE APÓS MITIGAÇÃO (Vppm)")
print("="*80)

vppm = calcular_vf(pre1, ifpp1, pre2, ifpp2, pre3, ifpp3)

print("-"*80)
print(f"\nVALOR FINANCEIRO DO RISCO POR PERDA DE PRODUTIVIDADE APÓS MITIGAÇÃO (Vppm): R$ {vppm:,.2f}")
print("="*80)

"""
Valor financeiro do risco por processo judicial (Vpj)
Vpj = Σ(Pr_i/100 × IFpj)
"""
# IFpj correspondente para cada agravo
ifpj1 = ifpj  # intoxicação exógena
ifpj2 = ifpj  # síndrome metabólica
ifpj3 = ifpj  # Alzheimer

# Cálculo de Vpp
print("="*80)
print(" CÁLCULO DO VALOR FINANCEIRO DO RISCO POR PROCERSSO JUDICIAL APÓS MITIGAÇÃO (Vpjm)")
print("="*80)

vpjm = calcular_vf(pre1, ifpj1, pre2, ifpj2, pre3, ifpj3)

print("-"*80)
print(f"\nVALOR FINANCEIRO DO RISCO POR PROCERSSO JUDICIAL APÓS MITIGAÇÃO (Vpjm): R$ {vpjm:,.2f}")
print("="*80)

# Cálculo do Valor fnanceiro do Risco após mitigação (Vm)
vm = calcular_vm(vppm, vpjm)

print("="*80)
print(" CÁLCULO DO VALOR FINANCEIRO DO RISCO APÓS MITIGAÇÃO (Vm)")
print("="*80)

print("-"*80)
print(f"\nVALOR FINANCEIRO DO RISCO APÓS MITIGAÇÃO (Vm): R$ {vm:,.2f}")
print("="*80)


# COMMAND ----------

# MAGIC %md
# MAGIC #### **MITIGAÇÃO DO RISCO**
# MAGIC
# MAGIC As ações de mitigação visam lidar com os impactos socioambientais do risco, de forma a reduzir a possibilidade de sua ocorrência.
# MAGIC
# MAGIC Para a demonstração ilustrativa da estrutura, foi elaborado um plano de ação simplificado para mitigar os riscos à saúde provocados pela deriva aérea como um processo de melhoria contínua baseado no ciclo PDCA (Planejar, Executar, Verificar e Agir) do Project Management Institute [PMI] (2021).
# MAGIC
# MAGIC Os recursos, quantidades dos recursos e os respectivos valores unitários foram definidos como parâmetros ilustrativos da demonstração. 
# MAGIC
# MAGIC **1 Premissas Adotadas**
# MAGIC
# MAGIC - Na ausência de dados do negócio específico de Belterra, os valores unitários foram obtidos por pesquisa exploratória de preços de mercado na internet, sem vínculo com fornecedores, e utilizados como referências substitutas dos custos efetivos de aquisição ou contratação.
# MAGIC
# MAGIC - Os sites consultados não são especificados no trabalho, uma vez que a identificação dos fornecedores não constitui objeto da pesquisa.
# MAGIC
# MAGIC **2 Ações de Mitigação Simplificadas**
# MAGIC - Planejamento meteorológico (planejamento): uso de modelos de previsão baseados na temperatura, umidade relativa, velocidade e direção do vento para programar as atividades de pulverização do glifosato quando as condições climáticas estiverem dentro de limites estabelecidos na literatura (Pellenz et al., 2026) para a redução da deriva;
# MAGIC - Aplicação da pulverização (execução): utilização de pontas de jato plano com indução de ar para gerar gotas maiores e mais pesadas, reduzindo a fração de gotas finas e, consequentemente, a deriva influenciada pelos fatores climáticos (Contiero et al., 2018);
# MAGIC - Monitoramento meteorológico (verificação e ação corretiva): medição das condições climáticas (temperatura, umidade relativa, velocidade e direção do vento) durante a atividade de pulverização por meio de uma estação meteorológica portátil acoplada ao veículo, permitindo a interrupção imediata da operação caso haja alteração das variáveis climáticas.
# MAGIC
# MAGIC
# MAGIC **3 Recursos para Implentar as Ações de Mitigação**
# MAGIC - Notebook;
# MAGIC - Estação meteorológica portátil;
# MAGIC - Plataforma web de previsão meteorológica;
# MAGIC - Ponta de jato plano com indução de ar;
# MAGIC - Consultoria agronômica;
# MAGIC - Treinamento em tecnologia de aplicação;
# MAGIC - Treinamento em monitoramento ativo durante a operação;
# MAGIC - Calibração técnica dos sensores da estação portátil.
# MAGIC
# MAGIC **4 Quantificação dos Recursos para Implemetar as Ações de Mitigação**
# MAGIC
# MAGIC CAPEX (Despesas de Capital)
# MAGIC
# MAGIC ![image_1790265062894.png](./image_1790265062894.png "image_1790265062894.png")
# MAGIC
# MAGIC OPEX (Despesas Operacionais)
# MAGIC
# MAGIC ![image_1790265086078.png](./image_1790265086078.png "image_1790265086078.png")
# MAGIC
# MAGIC **5 Custo da Ações de Mitigação do Risco**
# MAGIC
# MAGIC Os custos das ações de mitigação (Cm) são calculados na fórmula abaixo, e se tratam de uma simplificação do cálculo de fluxo de caixa citado por Garrison et al. (2013), pela soma direta dos custos de capital e operacionais:
# MAGIC
# MAGIC ![image_1790265123292.png](./image_1790265123292.png "image_1790265123292.png")
# MAGIC
# MAGIC em que: CAPEX é o investimento inicial de todos os recursos para implementar a ação; OPEX são os custos recorrentes de todos os recursos para manter a ação.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Custos da Mitigação do Risco
# ==========================================
# 1. INVESTIMENTO INICIAL (CAPEX)
# ==========================================
dados_capex = [
    {
        "Ação de Mitigação": "Planejamento Meteorológico", 
        "Descrição dos Recursos": "1 Notebook para acessar o serviço de previsão meteorológica", 
        "Valor Total (R$)": 7449.00
    },
    {
        "Ação de Mitigação": "Monitoramento Meteorológico", 
        "Descrição dos Recursos": "1 Estação meteorológica portátil para acoplar ao veículo agrícola", 
        "Valor Total (R$)": 10000.00
    }
]

df_capex = pd.DataFrame(dados_capex)
total_capex = df_capex["Valor Total (R$)"].sum()

# ==========================================
# 2. CUSTOS RECORRENTES (OPEX)
# ==========================================
# A coluna 'Quantidade Anual' atua como multiplicador para o cálculo automático
dados_opex = [
    {
        "Ação de Mitigação": "Planejamento Meteorológico", 
        "Descrição dos Recursos": "1 Plataforma Web que oferece a previsão das condições meteorológicas", 
        "Período": "Anual", 
        "Valor Unitário (R$)": 286.90, 
        "Qtd. Anual": 1
    },
    {
        "Ação de Mitigação": "Planejamento Meteorológico", 
        "Descrição dos Recursos": "1 Consultoria agronômica terceirizada para a interpretação dos modelos preditivos e emissão das ordens de serviço", 
        "Período": "Mensal", 
        "Valor Unitário (R$)": 2500.00, 
        "Qtd. Anual": 12
    },
    {
        "Ação de Mitigação": "Aplicação da Pulverização", 
        "Descrição dos Recursos": "40 Pontas de jato plano com indução de ar", 
        "Período": "Anual", 
        "Valor Unitário (R$)": 73.87, 
        "Qtd. Anual": 40
    },
    {
        "Ação de Mitigação": "Aplicação da Pulverização", 
        "Descrição dos Recursos": "2 Treinamentos de operadores de máquinas em tecnologia de aplicação", 
        "Período": "Anual", 
        "Valor Unitário (R$)": 800.00, 
        "Qtd. Anual": 2
    },
    {
        "Ação de Mitigação": "Monitoramento Meteorológico", 
        "Descrição dos Recursos": "1 Calibração técnica dos sensores da estação portátil", 
        "Período": "Semestral", 
        "Valor Unitário (R$)": 1000.00, 
        "Qtd. Anual": 2
    },
    {
        "Ação de Mitigação": "Monitoramento Meteorológico", 
        "Descrição dos Recursos": "2 Treinamentos de operadores de máquina para monitoramento ativo", 
        "Período": "Anual", 
        "Valor Unitário (R$)": 800.00, 
        "Qtd. Anual": 2
    }
]

df_opex = pd.DataFrame(dados_opex)

# Calculando o Valor Anual automaticamente
df_opex["Valor Anual (R$)"] = df_opex["Valor Unitário (R$)"] * df_opex["Qtd. Anual"]
total_opex = df_opex["Valor Anual (R$)"].sum()

# Calculando o custo da Mitigação do Risco
custo_mitigacao = total_capex + total_opex


# ==========================================
# 3. EXIBIÇÃO DOS DADOS
# ==========================================
print("="*60)
print(" TABELA C - INVESTIMENTO INICIAL (CAPEX)")
print("="*60)
# Formatando para exibição estilo moeda
display_capex = df_capex.copy()
display_capex["Valor Total (R$)"] = display_capex["Valor Total (R$)"].map("R$ {:,.2f}".format)
print(display_capex.drop(columns=["Ação de Mitigação"]).to_string(index=False))
print("-" * 60)
print(f"CUSTO TOTAL DE IMPLANTAÇÃO (CAPEX): R$ {total_capex:,.2f}")
print("\n")

print("="*80)
print(" TABELA D - CUSTOS RECORRENTES (OPEX)")
print("="*80)
display_opex = df_opex.copy()
display_opex["Valor Unitário (R$)"] = display_opex["Valor Unitário (R$)"].map("R$ {:,.2f}".format)
display_opex["Valor Anual (R$)"] = display_opex["Valor Anual (R$)"].map("R$ {:,.2f}".format)
# Exibe as colunas principais alinhadas
colunas_exibicao = ["Descrição dos Recursos", "Período", "Valor Unitário (R$)", "Valor Anual (R$)"]
print(display_opex[colunas_exibicao].to_string(index=False))
print("-" * 80)
print(f"CUSTO TOTAL DE MANUTENÇÃO (OPEX ANUAL): R$ {total_opex:,.2f}")

# Custo total da mitigação do risco
print("="*60)
print(" TABELA B - CUSTO TOTAL DE MITIGAÇÃO DE RISCOS")
print("="*60)
print(f"CUSTO TOTAL DE MITIGAÇÃO DE RISCOS: R$ {custo_mitigacao:,}")
print("\n")
print("NOTA: O custo total de mitigação de riscos é uma estimativa e pode variar de acordo com a implementação real.")
print("-" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC #### **ANÁLISE CUSTO BENEFÍCIO DA MITIGAÇÃO**
# MAGIC
# MAGIC A **análise custo-benefício da mitigação (ACBm)** é a economia gerada pela mitigação dos riscos, comparando os custos com e sem mitigação por meio da diferença entre o valor financeiro ponderado do risco (Vr) e a soma dos custos das ações de mitigação (Cm) com o valor financeiro ponderado do risco após a sua mitigação (Vm).
# MAGIC
# MAGIC ![image_1790272543668.png](./image_1790272543668.png "image_1790272543668.png")
# MAGIC

# COMMAND ----------

# DBTITLE 1,Cálculo da Análise Custo-Benefício
# Cálculo da Análise Custo-Benefício da Mitigação (ACBm)
# Fórmula: ACBm = VEr - (Cm + VEm)

# Vr: valor financeiro ponderado do risco sem mitigação (da célula 11)
# Vm: valor financeiro ponderado do risco após a sua mitigação (da célula 12)
# Cm: custo total de mitigação de riscos (da célula 14)

acbm = vr - (custo_mitigacao + vm)

print("="*60)
print(" ANÁLISE CUSTO-BENEFÍCIO DA MITIGAÇÃO (ACBm)")
print("="*60)
print(f"Valor financeiro ponderado do risco (Vr): R$ {vr:,.2f}")
print(f"Custo Total de Mitigação (Cm): R$ {custo_mitigacao:,.2f}")
print(f"Valor financeiro ponderado do risco após Mitigação (Vm): R$ {vm:,.2f}")
print("-"*60)
print(f"ANÁLISE CUSTO-BENEFÍCIO DA MITIGAÇÃO (ACBm): R$ {acbm:,.2f}")
print("="*60)

# COMMAND ----------

# MAGIC %md
# MAGIC #### **LUCRO BRUTO E LÍQUIDO DO NEGÓCIO**
# MAGIC
# MAGIC **1 Premissas Adotadas**
# MAGIC - Selecionou-se dados da tabela de lavouras temporárias da base do PAM (Pesquisa Agrícola Municipal), IBGE, do plantio de soja na cidade de Belterra e no ano de 2024 - ano mais recente da base.
# MAGIC - A receita bruta será calculada usando a área de plantação de 326,61 hectares, que se encontra a 87,70 metros de uma comunidade.
# MAGIC - Foram considerados como custos diretos da produção (Cd) na lavoura a mão-de-obra, fertilizante, agrotóxico e semente.
# MAGIC -  O valor da semente foi utilizado exclusivamente como referências para a demonstração, sem vínculo ou indicação de fornecedores. Os sites consultados não são especificados no trabalho, uma vez que a identificação dos fornecedores não constitui objeto da pesquisa.
# MAGIC - Na ausência de dados sobre condições de solo para a cálculo e da quantidade de uso de fertilizantes no estado do Pará, considerou-se 100 kg/ha de um produto NPK (nitrogênio, fósforo, potássio) vendido no estado do Pará segundo a CONAB, com valor médio calculado para os anos de 2023 a 2025.
# MAGIC - As informações sobre semeadura e quantidade de fertilizante por hectare foram atribuídos exclusivamente para ilustrar a demonstração, sem relação com plantações locais da região do conflito.
# MAGIC
# MAGIC **2 Lucro Bruto do Negócio (Lb)**
# MAGIC
# MAGIC O **lucro bruto do negócio (Lb)** é o lucro gerado exclusivamente pelas atividades-fim do negócio, sem considerar as receitas e despesas financeiras e os tributos, sendo baseada em parte da análise custo-volume-lucro (CVL) de Garrison et al. (2013). 
# MAGIC
# MAGIC ![image_1790275624023.png](./image_1790275624023.png "image_1790275624023.png")
# MAGIC
# MAGIC em que: Rv é a receita bruta de venda; Cd são os custos diretos da produção.
# MAGIC
# MAGIC **Receita bruta de venda (Rv)**
# MAGIC
# MAGIC A **receita bruta de venda (Rv)** é o valor total que se obtém com a venda da produção, sem deduzir impostos, fretes, descontos, dentre outros, sendo a a aplicação do conceito de venda da análise custo-volume-lucro (CVL) apresentada por Garrison et al. (2013).
# MAGIC
# MAGIC ![image_1790275646133.png](./image_1790275646133.png "image_1790275646133.png")
# MAGIC
# MAGIC em que: Qs é a quantidade de soja produzida na área de cultivo mais próxima da comunidade, em kg; Vs é o valor de venda da  soja, R$/kg.
# MAGIC
# MAGIC A **quantidade de soja produzida na área de cultivo (Qs)** é determinada pelo produto entre a quantidade de soja produzida por área (Qa), em kg/ha, a área plantada (Ap), em ha:
# MAGIC
# MAGIC ![image_1785011992596.png](./image_1785011992596.png "image_1785011992596.png")
# MAGIC
# MAGIC **Custos diretos da produção (Cd)**
# MAGIC
# MAGIC Os **custos diretos da produção (Cd)** são a soma de todos os custos que variam com a área plantada e estão diretamente ligados à produção, sendo criada a partir da ideia de custos primários (Garrison et al., 2013).
# MAGIC
# MAGIC ![image_1790275665956.png](./image_1790275665956.png "image_1790275665956.png")
# MAGIC
# MAGIC em que: CD são os valores de cada custo direto da produção, variando de 1 a c, como sementes, fertilizantes, agrotóxicos, mão de obra.
# MAGIC
# MAGIC
# MAGIC **3 Lucro Líquido do Negócio (L)**
# MAGIC
# MAGIC
# MAGIC O **lucro líquido do negócio sem sustentabilidade e com a ocorrência do risco (LSEMcr)** é calculado pela fórmula abaixo, sendo adaptada da análise custo-volume-lucro (CVL) de Garrison et al. (2013):
# MAGIC
# MAGIC ![image_1790275741263.png](./image_1790275741263.png "image_1790275741263.png")
# MAGIC
# MAGIC em que: Lb é o lucro bruto; Vr é o valor financeiro ponderado do risco.
# MAGIC
# MAGIC O **lucro líquido do negócio sem sustentabilidade e sem a ocorrência do risco (LSEMsr)** é calculado pela fórmula abaixo, sendo adaptada da análise custo-volume-lucro (CVL) de Garrison et al. (2013):
# MAGIC
# MAGIC ![image_1790275754894.png](./image_1790275754894.png "image_1790275754894.png")
# MAGIC
# MAGIC em que: Lb é o lucro bruto.
# MAGIC
# MAGIC O **lucro líquido do negócio com sustentabilidade e com a ocorrência do risco (LCOMcr)** está na fórmula abaixo, sendo adaptada da análise custo-volume-lucro (CVL) de Garrison et al. (2013):
# MAGIC
# MAGIC ![image_1790275795969.png](./image_1790275795969.png "image_1790275795969.png")
# MAGIC
# MAGIC em que: Lb é o lucro bruto; Cm é o custo da ação de mitigação; Vm é o valor financeiro ponderado do risco após a sua mitigação.
# MAGIC
# MAGIC O **lucro líquido do negócio com sustentabilidade e sem a ocorrência do risco (LCOMsr)** está na fórmula abaixo, sendo adaptada da análise custo-volume-lucro (CVL) de Garrison et al. (2013):
# MAGIC
# MAGIC ![image_1790275812292.png](./image_1790275812292.png "image_1790275812292.png")
# MAGIC
# MAGIC em que: Lb é o lucro bruto; Cm é o custo da ação de mitigação.

# COMMAND ----------

# DBTITLE 1,Cálculo do Lucro Bruto
"""
Quantidade de soja produzida na área de cultivo (Qs)
Qs = Qa * Ap

Qa: quantidade de soja produzida por área, em kg/ha, para o ano de 2024 (PAM)
Ap: área plantada, ha (área calculada no QGIS da comunidade a 87,70 metros da plantação)

Pesquisa Agrícola Municipal (PAM) - IBGE: Tabela de lavouras temporárias
Campos: ano, id_municipio, rendimento_medio_producao
Filtros: cidade Belterra, ano 2024
Fonte: https://basedosdados.org/dataset/fc403b40-a7e1-40e7-9efe-910847b45a69?table=1fd6598f-2dfa-4f26-94bb-0d6bd4a4c873
"""

# Cálculo da quantidade de soja produzida na área de cultivo
quantidade_area = 3000  # kg/ha [PAM]
area_plantada = 326.61

quantidade_soja = quantidade_area * area_plantada

print(f"Quantidade de soja produzida na área de cultivo (Qs): {quantidade_soja:,} kg")

"""
Receita bruta de venda (Rv)
Rv = Qs * Vs

Qs: quantidade de soja produzida na área de cultivo, em kg, em 2024
Vs: valor de venda da soja, R$/kg, em 2024

Pesquisa Agrícola Municipal (PAM) - IBGE: Tabela de lavouras temporárias
Campos: ano, id_municipio, valor_producao, quantidade_produzida
Filtros: cidade Belterra, ano 2024
Fonte: https://basedosdados.org/dataset/fc403b40-a7e1-40e7-9efe-910847b45a69?table=1fd6598f-2dfa-4f26-94bb-0d6bd4a4c873
"""

# Cálculo do valor de venda da soja
valor_producao = 176400000.00  # R$ [PAM]
quantidade_produzida = 88200000 # kg [PAM]

valor_venda = valor_producao / quantidade_produzida

print(f"Valor de venda da soja (Vs): R${valor_venda:.2f} por kg")

# Cálculo da receita bruta de venda
receita_bruta = quantidade_soja * valor_venda

print(f"Receita bruta de venda (Rv): R${receita_bruta:,.2f}")

"""
Custos diretos da produção (Cd)
Cd = Σ(CDi)

CD: valores de cada custo direto da produção (semente, mão-de-obra, fertilizante, agrotóxico).
"""

# Cálculo do custo das sementes
'''
Cálculo da semeadura: https://ainfo.cnptia.embrapa.br/digital/bitstream/item/27484/1/Plantio-semeadura.pdf
'''
valor_semente = 9.00            # R$/kg (valor de mercado)
quantidade_planta = 200000      # plantas por hectare (Cultivares de Soja)
espacamento_fila = 40           # centimetros (Cultivares de Soja)
germinacao_semente = 97         # porcentagem da germinação da semente, significa que a cada 100 sementes, 97 devem nascer (venda)
margem_seguranca = 10           # porcentagem extra que se adiciona para cobrir perdas de plântulas após a germinação
pms = 130                       # peso de mil sementes em gramas (Cultivares de Soja)

    # semente por hectare 
semente_ha = (quantidade_planta / (germinacao_semente / 100)) * (1 + (margem_seguranca / 100))

    # quilos de sementes por hectare (divide-se por 1.000.000 para converter gramas em quilogramas)
quilo_hectare = (semente_ha * pms) / 1000000

    # quantidade de sementes para 326.61 ha
quantidade_semente = quilo_hectare * area_plantada

    # custo das sementes
custo_semente = valor_semente * quantidade_semente

print(f"Custo da semente (CD1): R${custo_semente:,.2f}")

# Cálculo do custo da mão de obra (2 operarios)
# rendimento médio (célula 6) multiplicado pelo número de operários (célula 7) e pelo número de meses do ano
custo_mo = rendimento_medio * 2 * 12

print(f"Custo da mão-de-obra (CD2): R${custo_mo:,.2f}")

# Cálculo do custo do agrotóxico
'''
Companhia Nacional de Abastecimento (CONAB): Insumos Agropecuários
Filtros: Agrotóxico (grupo), hebicida (subgrupo), uf (PA), 2023 (ano de *), 2025 (ano até *)
Produto: GLIFOSATO, 480 G/L (CROTECT CROP SCIENCE LTDA - GLIFOSATO 480 SL ALAMOS)
Fonte: https://sistemas.conab.gov.br/consulta-precos-insumos/home
'''
valor_agro = 47.00     # média do valor, em R$, do litro comercializado entre os anos de 2023 e 2025 no Pará (CONAB)
litro_hectare = 6       # maior dose comercial do produto, em l/ha, obtido da bula do GLIFOSATO 480 SL ALAMOS

quantidade_agro = valor_agro * litro_hectare

custo_agro = quantidade_agro * area_plantada

print(f"Custo do agrotóxico (CD3): R${custo_agro:,.2f}")

# Cálculo do custo do fertilizante
'''
Companhia Nacional de Abastecimento (CONAB): Insumos Agropecuários
Filtros: Fertilizante (grupo), químico (subgrupo), uf (PA), 2023 (ano de *), 2025 (ano até *)
Produto: Produto: NPK (nitrogênio, fósforo e potássio) 10-28-20
Fonte: https://sistemas.conab.gov.br/consulta-precos-insumos/home

Adota-se o uso de 100 kg/ha
'''
valor_fertilizante = 241.66    # Valor médio do produto 10-28-20 no período para 50 kg

custo_fertilizante = (valor_fertilizante * 2) * area_plantada

print(f"Custo do fertilizante (CD4): R${custo_fertilizante:,.2f}")

# Cálculo do custo direto da produção (Pd)
custo_direto = custo_semente + custo_mo + custo_agro + custo_fertilizante

print(f"Custo direto da produção (Cd): R${custo_direto:,.2f}")

"""
Lucro bruto do negócio (Lb)
Lb = Rv - Cd

Rv: receita bruta de venda, R$
Cd: custos diretos da produção, R$
"""

lucro_bruto = receita_bruta - custo_direto

print(f"Lucro bruto do negócio (Lb): R${lucro_bruto:,.2f}")


# COMMAND ----------

# DBTITLE 1,Cálculo do Lucro Líquido
"""
lucro líquido do negócio sem sustentabilidade e com ocorrência do risco (LSEMcr)
Lsemcr = Lb - VEr

Lb: é o lucro bruto do negócio, R$ (da célula 18)
Vr: é o valor financeiro ponderado do risco, R$ (da célula 11)
"""

lsemcr = lucro_bruto - vr

print(f"Lucro líquido do negócio sem sustentabilidade e com ocorrência do risco (Lsemcr): R${lsemcr:,.2f}")

"""
lucro líquido do negócio sem sustentabilidade e sem ocorrência do risco (LSEMsr)
Lsemsr = Lb

Lb: é o lucro bruto do negócio, R$ (da célula 18)
"""

lsemsr = lucro_bruto

print(f"Lucro líquido do negócio sem sustentabilidade e sem ocorrência do risco (Lsemsr): R${lsemsr:,.2f}")

"""
lucro líquido do negócio com sustentabilidade e com ocorrência do risco (LCOMcr)
Lcomcr = Lb - Cm - VEm

Lb: é o lucro bruto do negócio, R$ (da célula 18)
Cm: é o custo da ação de mitigação, R$ (da célula 14)
Vm é o valor financeiro ponderado do risco após a sua mitigação, R$ (da célula 12)
"""

lcomcr = lucro_bruto - custo_mitigacao - vm

print(f"Lucro líquido do negócio com sustentabilidade e com ocorrência do risco (Lcomcr): R${lcomcr:,.2f}")

"""
lucro líquido do negócio com sustentabilidade e sem ocorrência do risco (LCOMsr)
Lcomsr = Lb - Cm

Lb: é o lucro bruto do negócio, R$ (da célula 18)
Cm: é o custo da ação de mitigação, R$ (da célula 14)
"""

lcomsr = lucro_bruto - custo_mitigacao

print(f"Lucro líquido do negócio com sustentabilidade e sem ocorrência do risco (Lcomsr): R${lcomsr:,.2f}")

