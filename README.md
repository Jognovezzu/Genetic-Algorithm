# Genetic-Algorithm
Work Project developed for the Discipline of Mechanics of Solids, using Genetic Algorithm to propose the best metallic structure (truss)


O objetivo principal do projeto é definido por buscar a melhor solução para a construção de uma estrutura em treliça howe, com base no comprimento das barras disponiveis e com a area da sessão transversal dessa barra, tentando buscar a estrutura com menor peso e menor Deformação da estrutura.

![treliça howe](https://user-images.githubusercontent.com/91996574/196968415-fbfb77b5-3861-4646-abb3-c74218720c8c.png)
\n
\n
Neste projeto apenas os comprimentos das barras da Base da Treliça (Barras 1,2,3,4) não podem ser alteradas, permanecendo em 2(m);
As principais alterações no comprimento das barras ocorre nas barras verticais (Barras 6, 8 e 10), podendo possuir os seguintes valores: 1m, 2m ou 3m.
Obs: As Barras em diagonais acompanham a mudança de comprimento das barras verticais.

\n\n

A segunda busca está na busca da melhor area da sessão transversal das barras. Onde cada barra pode possuir esta area com os seguintes valores: 3x10^{-3}m, 4x10^{-3}, 5x10^{-3}.


\n
\n

A saída do algoritmo é a melhor solução encontrada seguindo a Função Objetivo: ( 1 / ((0.1 * peso) + (Deformação no Ponto C)))
