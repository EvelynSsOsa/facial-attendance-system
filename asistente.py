import cv2
import face_recognition as fr
import os
import numpy as np
from datetime import datetime

## creando la base de datos
ruta = 'Empleados' ## esta sera el nombre de la carpeta, donde  tengo las fotos de los empleados
mis_imagenes = []## aqui vamos a ir cargando las imagenes que tenemos en la carpeta "Empleados" pero con la terminación ".jpg"
nombres_empleados = []## aqui vamos a obenter el nombre de las fotos, pero sin la terminación ".jpg"
lista_empleados = os.listdir(ruta) ## aqui vamos a obtener la dirección de la carpeta "Empleados", ya que a "os.listdir" le pasamos como parametro "ruta"

# decimos que por cada nombre(archivo de imagen) que encuentre en la carpeta de empleados
for nombre  in lista_empleados:
    ## cada una de las imagenes de la carpeta, la guardaremos en la variable "imagen_actual"
    ## vamos a utilizar el metodo ".imread" de cv2, que nos ayuda a leer cada una de las imagenes
    imagen_actual = cv2.imread(f'{ruta}/{nombre}') ## cosntruimos la ruta de donde lera las imagenes
    #                            carpta / nombre de cada imagen
    mis_imagenes.append(imagen_actual) ## ahora vamos a agregar a nuestra lista "mis.imagenes" el nombre de la imagen que obtuvgimos
    nombres_empleados. append(os.path.splitext(nombre)[0])
    ## ahora vamos a cargar unicamente el nombre de nuestros empleados a la lista "nombres_empleados", esto lo hacemos por metodo
    ##.path.splitext -> que le pasamos como parametro el nombre de la imagen y decimos que solo queremos la posición 0, osea lo que viene antes del punto


def codificar (imagenes):
    ## vamos a crear una lista vacia, que la vamos a llenar con
    ## las imagenes ya calculadas en vectores de acuerdo a sus caracteristicas
    lista_codificada = []

    ## pasar todas la imagenes a RGB por que "fr"(face_recognition) trabaja mejor con esos colores
    for imagen in imagenes:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB) ## lo hacemos por medio del metodo "cv2.cvtColor"
        ## donde le pasamos como parametro la imagen de la iteración actual y este otro "cv2.COLOR_BGR2RGB" que es la transición de un formato a otro

        ## CODIFICANDO:
        ## aqui vamos a codificar a cada una de las imagenes, es decir que vamos a calcular las carecteriticas faciales de cada uno de los ususarios metiendo
        ## a la imagen en vectores de 128 numeros
        codificando = fr.face_encodings(imagen)[0] ## con el 0 inidcamos que queremos reconocer unicamente el 1er rostro que veamos en la imagen

        ##AGREGAR LA IMAGEN CODIFICADA:
        ## en la lista "lista_codificada" pasamos la imagen codificada como parametro, que tenemos en la variable "codificado" linea 38
        lista_codificada.append(codificando)

    ##VAMOS A REGRESRA LA LISTA YA CON LAS IMAGENES CODIFICADAS
    return lista_codificada

## regustrar ingresos
def registrar_ingresos(persona): ## al crear a la función le pasamos como parametro una persona, que tambien puede ser un nombre
    f = open('registro.csv', 'r+') ## luego por medio del metodod "open" vamos a abrir el archivo de "registro.csv" en modo de lectura por "r+"
    lista_datos = f.readlines() ## ahora por medio del metodo "lista_datos" vamos a leer las lineas de ese archivo, es decir las columnas
    nombres_registro =[] ## creamos una lista vacia
    for linea in lista_datos: ## decimos que por cada linea en la "lista_datos"
        ingreso = linea.split(',') ## un ingreso sera separado por comas
        nombres_registro.append(ingreso[0])## ahora decimos que a la lista "nombres_registro" le vamos a agregar a "registro" en la 1era posición

    if persona not in nombres_registro: ## ahora decimos que si la "persona" no esta  "nombres_registro" (es decir que no se haya registrado antes)
        ahora = datetime.now() ## vamos a optener la fechaa actual
        string_ahora = ahora.strftime('%H:%M:%S') ## le aplicamos a la variable el metodo ".strftime" jusnto con el formato de la fecha
        f.writelines(f'\n{persona},{string_ahora}')## ahora vamos a escribir en esa linea, el nombre de la persona y la hora de entrada (string_ahora)


## vamos a crear una lista que nos ayuda, a guardar el resultado de llamar a la función "codificar" y le pasamos como
##parametro la lista "mis_imagenes" (esta lista se crea en la linea 9) -> es la encargada de contener a las imagenes
lista_empleados_codificada = codificar(mis_imagenes)
## como resultado en la lista "lista_empleados_codificada", tenemos los vectores
## de características faciales (128 valores) de cada empleado


#tomar imagen de camara web
captura = cv2.VideoCapture(0,cv2.CAP_DSHOW)

## Leer imagen de la camara
exito, imagen = captura.read()
##Se toma una fotografía desde la cámara web.
## "exito" devuelve True si la captura fue correcta.
## "imagen" almacena la captura

if not exito:
    print("No se a podido tomar la cpatura") ### en dado caso no haber podido tomar la captura, se mostrara este mensaje en consola
else:
    ## reconocer cara/ rostro en la captura, es decir obetenemos la ubicación del rostro/ cara
    cara_captura = fr.face_locations(imagen)
    ## codificar cara capturada
    cara_captura_codificada = fr.face_encodings(imagen, cara_captura)
    ## con ayuda del metodod ".face_encodings" pasamos la imagen y la ubicación de la imagen
    #se convierte en un vector
    ## numérico de 128 características faciales

    ## buscar concidencias:
    ## zip() permite recorrer simultáneamente dos listas:
    ## "cara_captura_codificada": contiene los vectores de 128 características
    ##de cada rostro detectado.
    ## "cara_captura": contiene las coordenadas de ubicación de cada rostro.
    ## Ahora, en cada iteración tenemos que :
    ## "cara_codif" almacena la codificación del rostro actual.
    ## "caraubic" almacena sus coordenadas dentro de la imagen.
    for cara_codif, caraubic in zip(cara_captura_codificada,cara_captura):
        coincidencia = fr.compare_faces(lista_empleados_codificada, cara_codif)
        ## Se compara el rostro capturado con todos los rostros, por medio del metodo ".compare_faces", le pasamos como paramtro la lista creada en la linea 64(lista_empleados_codificada)
        ##y "cara_codif" tomaría el valor del rostro actual codificado
        ## El resultado es una lista de valores booleanos (True o False) que indican si existe coincidencia.
        distancias = fr.face_distance(lista_empleados_codificada, cara_codif)
        ## ahora vamos a ver las similitudes entre rostros, por medio del metodo ".face_distance", le pasamos como parametro la lista que contiene todos los rostros de los empleados,
        ## pero ya codoficados, esa lista se crea en la linea 64 y la "cara_codif" que tomaria el valor del rostro actual (la persona que quiere ingresar)


        print(distancias) ## imprimimos en consola  las disntancias que hay entre cada uno de los empleados en ceunato ragos fisicos

        indice_coincidencia = np.argmin(distancias)
        ## np.argmin() devuelve la posición del valor
        ## más pequeño dentro del arreglo de distancias
        ## Ese índice corresponde al empleado más parecido.

        ## decimos que del arreglo "distancias" que fue creado en la linea 102, le pasamos como parametro el
        ##indice de coincidencia, es decir el indice mas pequeño de "distancia" es mayor a 0.6

        if distancias[indice_coincidencia] > 0.6:
            print('No coincide con ninguno de nuestros empleados') ## saldra en consola el siguiente mensaje

        else: ## en dado cadso de que haya un numero menor a 0.6
            ## buscar el nombre del empleado encontrado, de acuerdo al indice en la lista de "nombres_empleados"
            nombre = nombres_empleados[indice_coincidencia]

            y1,x2,y2,x1 = caraubic
            cv2.rectangle(imagen,(x1,y1),(x2,y2), (0,255,0), 2)## Se dibuja un rectángulo verde (0,255,0) alrededor del rostro, reconocido.
            cv2.rectangle(imagen, (x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            ## Se crea un rectángulo relleno donde
            ## se mostrará el nombre del empleado.
            cv2.putText(imagen, nombre, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255),2)
            ## Se escribe el nombre del empleado
            ## reconocido debajo del rostro detectado.

            registrar_ingresos(nombre)
            ##mandamos a llamar a la función "registrar_ingresosos" y le pasamos como parametro el nombre
            ## Se almacena el nombre y la hora actual
            ## en el archivo registro.csv
            ## mostrar la imagen contenida
            cv2.imshow('Imagen Web', imagen) ## Se almacena el nombre y la hora actual
            ## en el archivo registro.csv

            ## mantener ventana abierta
            cv2.waitKey(0)




