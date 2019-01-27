# Projeto Demonstrativo 1 - Princípios de Visão Computacional

## Organização dos arquivos do projeto
Um único arquivo foi desenvolvido, main.py. Este arquivo pode ser executado a partir da linha de comando com
```
python main.py --mode MODE_OPT
```
    
Em que `--mode` e `MODE-OPT` são argumentos para o programa que permitem selecionar um modo de execução. A fim de que fossem atendidos os 4 requisistos da aplicação, foram desenvolvidos 3 modos: modo imagem, modo vídeo e modo webcam.

O programa foi enviado com duas imagens, `image1.jpg` e `image2.jpg`, e um vídeo, `video1.mp4`.

A organização da pasta é a seguinte:
```
/trab1-pvc
    /images
        image1.jpg
        image2.jpg
    /videos
        video1.mp4
    main.py
```

## Modo de imagem (requisitos 1 e 2)
Para executar o modo de imagem do programa, basta entrar na pasta em que o arquivo está localizado e executar o comando:
```
python main.py --image IM_PATH
```
    
Em que `IM_PATH` é uma string tal como `images/image1.jpg` que indica o caminho da imagem no disco. Como exemplo, para abrir a imagem `image.jpg`, o comando é:
```
python main.py --image images/image1.jpg
```

## Modo de vídeo (requisito 3)
Para executar o modo de vídeo do programa, novamente basta entrar na pasta em que o arquivo está localizado e executar o comando:
```
python main.py --video VID_PATH
```

Tal como antes, `VID_PATH` é uma string que indica o caminho para o vídeo, tal como `videos/video1.mp4`. Para abrir o arquivo `video1.mp4`, por exemplo, temos:
```
python main.py --video videos/video1.mp4
```

## Modo de webcam (requisito 4)
Para executar o modo de vídeo do programa, novamente basta entrar na pasta em que o arquivo está localizado e executar o comando:
```
python main.py --webcam [CAM_DEV]
```
    
Neste caso, `CAM_DEV` é um inteiro que indica qual dispositivo de câmera utilizar, e é um argumento opcional. Caso não seja dado, será o usado o dispositivo 0, a câmera padrão. Como exemplo, para abrir o dispositivo de câmera 0, pode ser usado o comando:
```
python main.py --webcam 0
```

## Redimensionamento de imagens
No programa vem incluído um redimensionamento das imagens ou quadros de entrada, caso queria se utilizar imagens muito grandes ou diminuir o tamanho de vídeos para aumento de performance. Para utilizar o redimensionamento, basta incluir o argumento `--resize RSZ_STD` no comando, em que `RSZ_STD` é um dos padrões de qualidade implementados, que são o `qhd` (quasi-HD), `hd` (HD) e `fhd` (full-HD). Note que o redimensionador mantém o aspect ratio da imagem, mas garante que a imagem será sempre menor ou igual às dimensões do padrão dado. Note que este argumento pode ser utilizado nos três modos de execução. Para abrir a imagem `image2.jpg` usando o redimensionamento, por exemplo, pode ser usado o comando:
```
python main.py --image images/image2.jpg --resize qhd
```
