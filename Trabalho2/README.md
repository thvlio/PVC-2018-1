# Projeto Demonstrativo 2 - Princípios de Visão Computacional

## Organização dos arquivos do projeto
Um arquivo principal foi desenvolvido, main.py. Este arquivo pode ser executado a partir da linha de comando com
    python main.py --mode MODE_OPT --smode SMODE_OPT
    
Em que --mode e MODE-OPT e --smode e SMODE_OPT são argumentos para o programa que permitem selecionar um modo de execução. A fim de que fossem atendidos os 4 requisistos da aplicação, foram desenvolvidos 4 modos: modo line, modo intrinsic, modo extrinsic, modo ruler.

## Modo line (requisitos 1)
Ex:
    python main.py --line --image images/intset/set1/img32.png

## Modo intrinsic (requisitos 2)
Ex:
    python main.py --intrinsic --image images/intset/set4/img665.jpg --xml xml --imageset images/intset

## Modo extrinsic (requisito 3)
Ex:
    python main.py --extrinsic --xml xml --imageset images/extset

## Modo ruler (requisito 4)
Ex:
    python main.py --ruler --xml xml --image images/extset/set2/img5.png --imageset dmed_2
    
Nota: repositório e código precisam de limpeza

Imagens de teste: https://www.dropbox.com/sh/gb3rknfxfij8duu/AABebPVnXN_djABVtwX4MSlEa?dl=0
