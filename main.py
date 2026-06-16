from cv2 import cv2 ## esta libreria la he utilizdo antes en otros proyectos para treabajr con la imagenes
import face_recognition as fr ## esta nos srive para poder trabajr con los rostros de las personas, su alias es "fr"

## cargar imagenes
foto_control = fr.load_image_file('FotoA.jpg')
foto_prueba = fr.load_image_file('FotoB.jpng')

## Pasar imagenes a RGF, porque la librería de "face_recognition" trabaja mejor así
foto_control = cv2.cvtColor(foto_control, cv2.COLOR_BGR2RGB)
foto_prueba = cv2.cvtColor(foto_prueba, cv2.COLOR_BGR2RGB)

##localizar cara control:
## tenemos que con fr y el metodo "face_locations" nos ayuda a buscar las caras encontradas
## nos devuelve algo como lo siguinete "[(100,300,300,100)]" donde "(top, right, bottom, left)"
lugar_cara_A =fr.face_locations(foto_control)[0] ## se coloca el indice 0, poruqe solo queremos la 1era cara detectada

cara_codificada_A= fr.face_encodings(foto_control)[0]## la librreia "fr" convierte el rostro en un vector 128 números
## estos numeros representan las diferentes caracteristicas de un rostro, no son los ojos, ni la nariz, son medidas matematicas
## que nos van a servir para comparar rostros

##localizar cara control -> paso lo mismo que con la imagen A
lugar_cara_B=fr.face_locations(foto_prueba)[0] ## ->Aquí localizamos el rostro en la imagen
cara_codificada_B= fr.face_encodings(foto_prueba)[0] ## y aquí el vector con los 128 números, el que nos ayuda a sacar las caracteriticas

# mostrar rectangulos
## tomando en cuenta que tenemos que " (top, right, bottom, left)" en la variable "lugar cara "
                                      #  0,   1,      2,     3
cv2.rectangle(foto_control,
              (lugar_cara_A[3], lugar_cara_A[0]),
              (lugar_cara_A[1], lugar_cara_A[2]),
              (0,255,0),
              2)
## por medio de la libreria con alias "cv2"... indicamos que vamos a dibujar un rectangulo con "cv2.rectangle" y pasamos como paramtero
## la foto en la que queremos dibujar el rectangulo, en seguida las "cordenadas"
## (lugar_cara_A[3] - > left (que se encuentra en la 3era posisción) - > lado ixquierdo del rectangulo
## lugar_cara_A[0]) - > top (que es la posición 0) -> la parte de arriba del rectangulo
## (lugar_cara_A[1] -> right (que es la posisicón 1) -> la parte derecha del rectangulo
## lugar_cara_A[2]) - > botton (que es la posisción 2) -> es el fondo/ la parte de abajo del rectangulo
## (0,255,0), - > Con esto decimos que serpa de color verde
## y que el grosor será  de tamaño 2

## pasa exacatamente lo mismo para este otor rectangulo
cv2.rectangle(foto_prueba,
              (lugar_cara_B[3], lugar_cara_B[0]),
              (lugar_cara_B[1], lugar_cara_B[2]),
              (0,255,0),
              2)

## REALIZAR LA COMPARACIÓN:
## por medio de la libreeria con alias "fr" madamos a llamar al metodo de ".compare_faces" y le pasamos como paramtero
##las varaobales "cara_codificada_A" y "cara_codificada_B" -> Que si bien recordamos son la variables , que nos ayudaban a obtener las caracteristicas del rostro vectores de 128 numeros
## linea 22 y linea 17
resultado = fr.compare_faces([cara_codificada_A], cara_codificada_B) ## con esta linea solo obtener un "True" o "False"... pero no que tan diferentes son los rostros
## entonces si obtenemos un "True" -> probablemente sea la misma persona
## entonces si obtenemos un "False" -> probablemente son personas diferentes

## medida de la distancia:
## Aquí vamos a calcular que tan diferentes son un rostro del otro, la distancia que hay entre ellos de acuerdo a los vactores, que pasamos como parametros en las variables
## "cara_codificada_A" y "cara_codificada_B"... estos parametrso se los pasamos al metodo "face_distance" de la libreria con alias "fr"
distancia = fr.face_distance([cara_codificada_A], cara_codificada_B)

## vamos a mostrar el resultado en la foto de prueba
## para ello mandamos a llamar a la libreria por su alias "cv2", con el metodo "putText" a este metodo le vamos a pasar
## como paramtero la "foto_prueba" y con una cadena de texto pasasmo los paramtero/ variables donde tenermos guardados la distancia entre vectores "distancia" y "resultado"
## colocamos en que posisción donde queremos el texto con "(50,50)" y pasamos la fuente con la que queremos ver los resultados,
## el color con "(0,255,0)" y el grosor con el 2
cv2.putText(foto_prueba,
            f'{resultado} {distancia.round(2)}',
            (50,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2)

## mostrar imagenes con ".imshow"
cv2.imshow('Foto Control',foto_control )
cv2.imshow('Foto Prueba', foto_prueba)



## mantener el programa abierto
cv2.waitkey(0)