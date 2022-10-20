import numpy as np
import pandas as pd
import math
import random

def delta_f (comprimentos,forcas,area,forcash): #return (deltaC,seg,fN)
    mE = 2*(10**8)

    #Reações (Forças externas)
    #rAx = forcash[0] + forcash[1] + forcash[2] + forcash[3] + forcash[4]
    rAx = -(forcash[0] + forcash[1] + forcash[2] + forcash[3] + forcash[4])
    rEy = ((forcas[1]*comprimentos[0] + forcas[2]*(comprimentos[0]+comprimentos[1]) + forcas[3]*(comprimentos[0]+comprimentos[1]+comprimentos[2])+forcas[4]*(comprimentos[0]+comprimentos[1]+comprimentos[2]+comprimentos[3])) +( forcash[1]*comprimentos[5] + forcash[2]*comprimentos[7] + forcash[3]*comprimentos[9] )) / (comprimentos[0]+comprimentos[1]+comprimentos[2]+comprimentos[3]) 

    rAy = forcas[0] + forcas[1] + forcas[2] + forcas[3] + forcas[4] - rEy 
    
    #Cálculo das forças axiais nas barras - forças internas

    fN = [None]*13

    alpha = math.atan(comprimentos[5]/comprimentos[0])                                          # Angulo entre barra 1 e barra 6
    fN[4] = (forcas[0] - rAy) / math.sin(alpha)
    fN[0] = -rAx - fN[4]*math.cos(alpha)
    fN[1] = fN[0]
    fN[5] = 0

    beta = math.atan(comprimentos[0]/comprimentos[5])                                           # Angulo entre barra 5 e barra 6
    ipsolum = math.atan(comprimentos[1]/comprimentos[5])                                        # Angulo entre barra 6 e barra 7
    thetta = math.atan((comprimentos[7]-comprimentos[5])/comprimentos[1])                       # Angulo entre barra 12 e barra 7

    
    fN[6] = - ((forcas[1]*math.cos(thetta) + math.cos(beta)*math.cos(thetta)*fN[4]) + math.cos(thetta)*fN[5] - fN[4]*math.sin(beta)*math.sin(thetta))/(math.cos(ipsolum)*math.cos(thetta) + math.sin(ipsolum)*math.sin(thetta))
    if thetta==0:
        fN[11] = (fN[6]*math.cos(ipsolum) + fN[4]*math.cos(beta) + fN[5] + forcas[1])
    else:
        fN[11] = (fN[6]*math.cos(ipsolum) + fN[4]*math.cos(beta) + fN[5] + forcas[1])/  math.sin(thetta)
        

        
    dif = (comprimentos[7] - comprimentos[5])
    if dif ==0:
        roh = 0.90
    else:
        roh = math.atan(comprimentos[1] / dif)                                                  # Angulo entre barra 12 e barra 8

    dif = (comprimentos[7] - comprimentos[9])

    if dif ==0:
        romeu = 0.90
    else:
        romeu = math.atan(comprimentos[2] / dif)                                         # Angulo entre barra 8 e barra 13

    if romeu ==0:
        fN[12] = (fN[11]*math.sin(roh)) 
    else:
        fN[12] = (fN[11]*math.sin(roh)) / math.sin(romeu)

    fN[7] = - fN[11]*math.cos(roh) - fN[12]*math.cos(romeu) - forcas[2]

    a_w = math.atan(comprimentos[5]/comprimentos[1])                                            # Angulo entre barra 2 e barra 7
    a_i = math.atan(comprimentos[1]/comprimentos[5])                                            # Angulo entre barra 7 e barra 8
    a_6 = math.atan(comprimentos[2]/comprimentos[9])                                            # Angulo entre barra 8 e barra 9
    a_u = math.atan(comprimentos[9]/comprimentos[2])                                            # Angulo entre barra 9 e barra 3

    if a_6==0:
        fN[8] = (-fN[6]*math.cos(a_i) - fN[7])
    else:
        fN[8] = (-fN[6]*math.cos(a_i) - fN[7]) / math.cos(a_6)

        
    fN[2] = (fN[1] + fN[6]*math.cos(a_w) - fN[8]*math.cos(a_u))

    alpha1 = math.atan(comprimentos[0]/comprimentos[3])                                         # Angulo entre barra 4 e barra 11
    
    fN[10] = (forcas[4] - rEy) / math.sin(alpha)
    fN[3] = -rAx - fN[10]*math.cos(alpha)
    fN[9] = 0

    #forças virtuais - Deslocamento no nó C

    #Reações nos apoios
    fnC = 1
    rnAx = 0
    rnEy = fnC * (comprimentos[0] + comprimentos[1])/(comprimentos[0] + comprimentos[1] + comprimentos[2] + comprimentos[3])
    rnAy = fnC - rnEy

    forcas1 = [0]*5 #Por nao haver forças externas
    fNV = [None]*13
    fNV[4] = (- rnAy) / math.sin(alpha)
    fNV[0] = -rnAx - fNV[4]*math.cos(alpha)


    fNV[1] = fNV[0]
    fNV[5] = 0

    if ipsolum==0:
        fNV[6] = - ((forcas1[1]*math.cos(thetta) + math.cos(beta)*math.cos(thetta)*fNV[4]) + math.cos(thetta)*fNV[5] - fNV[4]*math.sin(beta)*math.sin(thetta))
    else:
        fNV[6] = - ((forcas1[1]*math.cos(thetta) + math.cos(beta)*math.cos(thetta)*fNV[4]) + math.cos(thetta)*fNV[5] - fNV[4]*math.sin(beta)*math.sin(thetta))/(math.cos(ipsolum)*math.cos(thetta) + math.sin(ipsolum)*math.sin(thetta))
    
    
    if thetta==0:
        fNV[11] = (fNV[6]*math.cos(ipsolum) + fNV[4]*math.cos(beta) + fNV[5] + forcas1[1])
    else:
        fNV[11] = (fNV[6]*math.cos(ipsolum) + fNV[4]*math.cos(beta) + fNV[5] + forcas1[1]) / math.sin(thetta)


    if romeu ==0:

        fNV[12] = (fNV[11]*math.sin(roh))
    else:
        fNV[12] = (fNV[11]*math.sin(roh)) / math.sin(romeu)

        
    fNV[7] = - fNV[11]*math.cos(roh) - fNV[12]*math.cos(romeu) - forcas1[2]


    if a_6==0:
        fNV[8] = (-fNV[6]*math.cos(a_i) - fNV[7] + fnC )
    else:
        fNV[8] = (-fNV[6]*math.cos(a_i) - fNV[7] + fnC )/ math.cos(a_6)


    fNV[2] = (fNV[1] + fNV[6]*math.cos(a_w) - fNV[8]*math.cos(a_u))


    fNV[10] = (forcas1[4] - rnEy) / math.sin(alpha)
    fNV[3] = -rnAx - fNV[10]*math.cos(alpha)
    fNV[9] = 0

    
    delta = []
        # CALCULO DO DELTA (Δ)
    for i in range(0,len(fN)):
        delta_v = (fN[i]*fNV[i]*comprimentos[i]) / (mE*area[i])
        delta.append(delta_v)

    # Arredondamentos
    fN = [float(round(item, 2)) for item in fN]
    fNV = [float(round(item, 2)) for item in fNV]
    comprimentos = [float(round(item, 2)) for item in comprimentos]
    delta1 = [format(item,'f') for item in delta]


    tabela = pd.DataFrame({'L (m)':comprimentos ,
                   'N (kN)': fN ,
                   'Ñ (kN)': fNV,
                   'Δ (m)':delta1})


    #print(tabela)
    #print('Total:                     ',round(sum(delta),6))
    deltaC = sum(delta)
    deltaC = round(deltaC,6)
    seg = seguranca(fN,area)
    return (deltaC,seg,fN)

def seguranca(forcas,areas): # return invalido
    invalido = False
    #forcas - kN -> 1000Pa
    #areas - m^2
    sigmaE = 300 #MPa ->300.000.000Pa
    coeficiente = 2
    raz = sigmaE/coeficiente # 150MPa -> 150.000.000Pa
    sigma = []
    for i in range(0,len(forcas)):
        aux = forcas[i]/areas[i]
        sigma.append(aux*0.001) #conversão kN/m² para MPa

    if max(sigma)>150:
    #or min(sigma)<-150:
        invalido = True

    
    return invalido

def populacao_aleatoria(n):
    areas = [3*10**-4,4*10**-4,5*10**-4]
    comprimentos_op = [1,2,3]

    """
    Argumentos da Função:
        n: Número de indivíduos
    Saída:
        Uma população aleatória. População é uma lista de indivíduos,
        e cada indivíduo é uma matriz 2x13 de pesos (números).
        [
            [[comprimento],[areas]]
            [[comprimento],[areas]]
            [[comprimento],[areas]]
        ]
    """
    populacao = []

    for i in range(n):
        comprimento = [2,2,2,2,None,None,None,None,None,None,None,None,None]
        comprimento_aux = np.random.choice(comprimentos_op,3)
        comprimento[5],comprimento[7],comprimento[9] = comprimento_aux[0],comprimento_aux[1],comprimento_aux[2]
        comprimento[4] = round(((comprimento[0]**2)+(comprimento[5]**2))**(1/2),4)
        comprimento[6] = round(((comprimento[1]**2)+(comprimento[5]**2))**(1/2),4)
        comprimento[8] = round(((comprimento[2]**2)+(comprimento[9]**2))**(1/2),4)
        comprimento[10] = round(((comprimento[3]**2)+(comprimento[9]**2))**(1/2),4)
        comprimento[11] = round((((comprimento[5]-comprimento[7])**2)+(comprimento[1]**2))**(1/2),4)
        comprimento[12] = round((((comprimento[9]-comprimento[7])**2)+(comprimento[1]**2))**(1/2),4)

        ar = np.random.choice(areas,13)
        individuo = [comprimento,ar]
        populacao.append(individuo)
    #print (populacao)
    return populacao



def crossover(individuo1, individuo2):
    CHANCE_CO = 0.6
    """
    Argumentos da Função:
        individuoX: matriz 2x13 com os pesos do individuoX.
    Saída:
        Um novo indivíduo com pesos que podem vir do `individuo1`
        (com chance 1-CHANCE_CO) ou do `individuo2` (com chance CHANCE_CO),
        ou seja, é um cruzamento entre os dois indivíduos. Você também pode pensar
        que essa função cria uma cópia do `individuo1`, mas com chance CHANCE_CO,
        copia os respectivos pesos do `individuo2`.
    """
    filho = individuo1.copy()
    for j in range(2):
        for i in range(13):
            if np.random.uniform(0, 1) < CHANCE_CO:
                filho[j][i] = individuo2[j][i]
    return filho


def ordenar_lista(lista, ordenacao, decrescente=True):
    """
    Argumentos da Função:
        lista: lista de números a ser ordenada.
        ordenacao: lista auxiliar de números que define a prioridade da
        ordenação.
        decrescente: variável booleana para definir se a lista `ordenacao`
        deve ser ordenada em ordem crescente ou decrescente.
    Saída:
        Uma lista com o conteúdo de `lista` ordenada com base em `ordenacao`.
    Por exemplo,
        ordenar_lista(['a', 'b', 'c', 'd'], [7, 2, 5, 4])
        # retorna ['a', 'c', 'd', 'b'] (o maior número é 7, que corresponde à primeira letra: 'a')
        ordenar_lista(['w', 'x', 'y', 'z'], [3, 8, 2, 1])
        # retorna ['x', 'w', 'y', 'z'] (o maior número é 8, que corresponde à segunda letra: 'x')
    """
    return [x for _, x in sorted(zip(ordenacao, lista), key=lambda p: p[0], reverse=decrescente)]


def proxima_geracao(populacao, fitness):
    NUM_MELHORES = 1
    """
    Argumentos da Função:
        populacao: lista de indivíduos.
        fitness: lista de fitness, uma para cada indivíduo.
    Saída:
        A próxima geração com base na população atual.
        Para criar a próxima geração, segue-se o seguinte algoritmo:
          1. Colocar os melhores indivíduos da geração atual na próxima geração.
          2. Até que a população esteja completa:
             2.1. Escolher aleatoriamente dois indivíduos da geração atual.
             2.2. Criar um novo indivíduo a partir desses dois indivíduos usando
                  crossing over.
             2.3. Mutar esse indivíduo.
             2.4. Adicionar esse indivíduo na próxima geração
    """
    # Adicionar os melhores indivíduos da geração atual na próxima geração
    ordenados = ordenar_lista(populacao, fitness)
    
    proxima_ger = ordenados[:NUM_MELHORES]

    while len(proxima_ger) < NUM_INDIVIDUOS:
        # Você pode usar a função random.choices(populacao, weights=None, k=2) para selecionar `k`
        # elementos aleatórios da população.
        #
        # Se vc passar o argumento `weights`, os indivíduos serão escolhidos com base nos pesos
        # especificados (elementos com pesos maiores são escolhidos mais frequentemente).
        # Uma ideia seria usar o fitness como peso.
        ind1,ind2 = random.choices(populacao, k=2)
        filho = crossover(ind1, ind2)
        #mutacao(filho)

        proxima_ger.append(filho)

    return proxima_ger

def mutacao(ind1):
    CHANCE_MU = 0.1

    filho = ind1.copy()
    for j in range(13):
        if np.random.uniform(0, 1) < CHANCE_MU:
            if filho[1][j] == (3*10**-4):
                aux = [4*10**-4,5*10**-4]
                filho[1][j] = np.random.choice(aux)
            if filho[1][j] == (4*10**-4):
                aux = [3*10**-4,5*10**-4]
                filho[1][j] = np.random.choice(aux)
            if filho[1][j] == (5*10**-4):
                aux = [4*10**-4,3*10**-4]
                filho[1][j] = np.random.choice(aux)
    
    comp_aux = [filho[0][5],filho[0][7],filho[0][9]]

    for i in range(3):
        if comp_aux[i]==1:
            aux = [2,3]
            comp_aux[i] = np.random.choice(aux)
        if comp_aux[i]==2:
            aux = [3,1]
            comp_aux[i] = np.random.choice(aux)
        if comp_aux[i]==3:
            aux = [2,1]
            comp_aux[i] = np.random.choice(aux)
    
    filho[0][5],filho[0][7],filho[0][9] = comp_aux[0],comp_aux[1],comp_aux[2]
    
    return filho


def calcular_fitness(ind):
    forcas = [0,55,20,40,10]
    forcash = [0,0,0,0,0]
    comprimentos = ind[0]
    areas = ind[1]
    deltaC = delta_f(comprimentos,forcas,areas,forcash)


    peso = 0 
    for i in range (0,len(comprimentos)):
        peso += areas[i]*comprimentos[i]
        
    peso = peso*7870   
    if deltaC[1] == True:
            return(0)
    else:
        return (1 / ((0.1*peso) + deltaC[0]))




num_geracoes = 100
NUM_INDIVIDUOS = 100

# Crie a população usando populacao_aleatoria(NUM_INDIVIDUOS)
populacao = populacao_aleatoria(NUM_INDIVIDUOS)
print('ger | fitness\n----+-' + '-'*5*NUM_INDIVIDUOS)
fitness_hist = []
for ger in range(num_geracoes):
# Crie uma lista `fitness` com o fitness de cada indivíduo da população
# (usando a função calcular_fitness e um `for` loop).
    fitness = []
    for ind in populacao:
        foo = calcular_fitness(ind)
        fitness.append(foo)
    aux = fitness
    aux.sort(reverse=True)
    fitness_hist.append(aux[0])
    # Atualize a população usando a função próxima_geração.
    populacao = proxima_geracao(populacao, fitness)
    '''
    print('{:3} |'.format(ger),
            ' '.join(f'{s}'.format(s) for s in sorted(fitness, reverse=True)))

# Opcional: parar se o fitness estiver acima de algum valor (p.ex. 300)
# if max(fitness) > 300:
#     break

# Calcule a lista de fitness para a última geração
'''
fitness = []
for ind in populacao:
    fitness.append(calcular_fitness(ind))

# Mostre o melhor indivíduo
ordenados = ordenar_lista(populacao, fitness)
melhor = ordenados[0]
print('Melhor individuo:', melhor)

#print(populacao,fitness)


forcas = [0,55,20,40,10]
forcash = [0,0,0,0,0]
comprimentos = melhor[0]
areas = melhor[1]

deltaC = delta_f(comprimentos,forcas,areas,forcash)


peso = 0 
for i in range (0,len(comprimentos)):
    peso += areas[i]*comprimentos[i]

peso = peso*7870

print(f'Deformação no Ponto C:{deltaC[0]*100}mm;\n Massa: {peso}Kg;\n Areas: {melhor[1]}\n Barra 6:{melhor[0][5]}m, \n Barra 8: {melhor[0][7]}m, \n Barra 10:{melhor[0][9]}m')

'''from matplotlib import pyplot as plt

plt.plot(range(len(fitness_hist)),fitness_hist)
plt.grid(True,zorder=0)
plt.title('Histórico')
plt.xlabel("Geração")
plt.ylabel("Fitness")
plt.show()
'''






