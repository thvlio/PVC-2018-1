# Projeto Demonstrativo 3 - Princípios de Visão Computacional

## Organização dos arquivos do projeto
Para testar o projeto, a pasta imgs/ deve ser baixada do seguinte link:
https://www.dropbox.com/sh/phxtg4fpoul2y6e/AADeLUtSmXNE6onY2CSYb0WHa?dl=0

Basta colocá-la na pasta raiz junto aos arquivos do projeto. Nela, estão contida as imagens capturadas e várias imagens descartadas ou de teste que podem servir para reproduzir alguns dos resultados mencionados no relatório.

Foram preparados três scripts para testar cada requisito: `req1.sh`, `req2.sh` e `req3.sh`. Recomendo apenas alterar o argumento `--scale` para algo como 0.2, pois a execução do template matching é muito lenta.

Lembre-se que antes de executar scripts, você deve rodar o comando:
```
chmod +x reqN.sh
```

## Requisito 1
O script 1, por padrão, gera os mapas para as imagens fornecidas, mostra na tela os mapas e salva no disco. Os comandos comentados são os usados para gerar a disparidade com o SGBM.

## Requisito 2
O script 2, por padrão, calcula a retificação necessária para as imagens e repete o requisito anterior. Novamente, a parte comentada corresponde a usar o SGBM para gerar a disparidade. Não recomendo alterar o descritor de SIFT para ORB, porque o descritor ORB se aglomera facilmente nas regiões onde há fortes features, e isso prejuda a retificação.
Caso queira trocar mesmo assim, isso pode ser feito trocando `--desc sift-1000` por `--desc orb-2000`, em que o numero depois do hifen é um limite superior no número de features detectados.

## Requisito 3
O script 3, por padrão, abre as duas imagens para o usuário e permite que seja clicado em pontos das imagens para medir alguma distância. Como ele precisa retificar as imagens para calcular a disparidade, ele usa o descritor SIFT.
