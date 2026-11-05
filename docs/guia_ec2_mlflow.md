# Guía: servidor de MLflow en AWS EC2 y experimentos de la Entrega 2

La entrega pide pantallazos de los experimentos registrados en MLflow **en una máquina EC2**,
donde se vean el usuario y la IP de la máquina, y la IP en MLflow. Al final, la máquina se
**detiene, no se termina**.

## 1. Crear (o reutilizar) la instancia

En AWS Academy: **Start Lab**, esperar el punto verde y entrar a **AWS → EC2 → Launch instance**.

| Campo | Valor |
|---|---|
| Nombre | `pgcb-mlflow` |
| AMI | Amazon Linux 2023 |
| Tipo | `t3.medium` (2 vCPU, 4 GB). Con `t2.micro` LightGBM se queda sin memoria |
| Llave | `Despliegue` (la misma `.pem` de los talleres) |
| Almacenamiento | 20 GB |
| IAM instance profile (*Advanced details*) | `LabInstanceProfile`: permite hacer `dvc pull` desde S3 sin copiar credenciales |

**Security group, reglas de entrada:**

| Tipo | Puerto | Origen | Para qué |
|---|---|---|---|
| SSH | 22 | My IP | Conectarse por terminal |
| Custom TCP | 5000 | 0.0.0.0/0 | Interfaz y API de MLflow |
| Custom TCP | 8050 | 0.0.0.0/0 | Tablero (opcional para esta entrega) |

Anote la **Public IPv4 address**, que cambia cada vez que se reinicia el lab.

## 2. Conectarse e instalar

```bash
chmod 400 ~/Documents/Despliegue/Despliegue.pem
ssh -i ~/Documents/Despliegue/Despliegue.pem ec2-user@<IP-PUBLICA>
```

Dentro de la máquina (MLflow 3 necesita Python ≥ 3.10; Amazon Linux trae 3.9, por eso se instala 3.11):

```bash
sudo dnf install -y python3.11 python3.11-pip git tmux libgomp
git clone https://github.com/melissagutierrezarias-create/pgcb-demanda.git
cd pgcb-demanda
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Memoria de respaldo (swap).** La t3.medium tiene 4 GB de RAM; con MLflow corriendo, los Random forest
la agotan y Linux mata el proceso (`Killed`). Se agregan 4 GB de swap una sola vez:

```bash
sudo dd if=/dev/zero of=/swapfile bs=128M count=32
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
free -h          # la fila Swap debe decir ~4.0Gi
```

El swap se apaga al reiniciar la máquina; al volver a encenderla basta con `sudo swapon /swapfile`.

**Datos.** Opción A, con `LabInstanceProfile`:

```bash
dvc pull data/processed/pgcb_horario_limpio.csv.dvc
```

Opción B, si `dvc pull` da error de permisos: desde **su Mac**, en otra terminal, copie el archivo:

```bash
scp -i ~/Documents/Despliegue/Despliegue.pem data/processed/pgcb_horario_limpio.csv \
    ec2-user@<IP-PUBLICA>:~/pgcb-demanda/data/processed/
```

## 3. Levantar el servidor de MLflow

Se usa `tmux` para que el servidor siga corriendo aunque se cierre la terminal:

```bash
tmux new -s mlflow
cd ~/pgcb-demanda && source .venv/bin/activate
mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts \
  --allowed-hosts "*" \
  --cors-allowed-origins "http://<IP-PUBLICA>:5000" \
  --workers 2
```

Salga de tmux sin detener el servidor con **Ctrl+B y luego D**.

`--allowed-hosts "*"` y `--cors-allowed-origins` son necesarios: MLflow 3 rechaza por defecto las
peticiones que llegan con la IP pública (en el navegador sale *Request error / INTERNAL_ERROR* y en el log
*Blocked cross-origin request* con `403 Forbidden`). Como la IP cambia cada vez que se enciende la
máquina, hay que poner la IP nueva en `--cors-allowed-origins`. `--workers 2` reduce la memoria que usa
el servidor. `sqlite` habilita el **Model Registry**, donde quedan registrados los modelos finales.

Compruebe en su navegador: `http://<IP-PUBLICA>:5000` debe mostrar MLflow.

## 4. Correr los experimentos

En la misma máquina:

```bash
cd ~/pgcb-demanda && source .venv/bin/activate
export MLFLOW_TRACKING_URI=http://localhost:5000
python -m src.experimentos     # 27 corridas de demanda + modelo final (≈ 5-8 min en t3.medium)
```

**Cada integrante puede registrar sus experimentos desde su computador** apuntando al mismo servidor;
sus corridas quedan en el MLflow de EC2. Por ejemplo:

```bash
export MLFLOW_TRACKING_URI=http://<IP-PUBLICA>:5000      # Windows PowerShell: $env:MLFLOW_TRACKING_URI="http://<IP-PUBLICA>:5000"
python -m src.entrenar --modelo lightgbm --conjunto basico_festivos --objetivo relativo --params '{"num_leaves": 31}'
```

## 5. Pantallazos para la entrega

En todos debe verse la IP. Tómelos con la barra de direcciones del navegador visible.

1. **Consola EC2**: la instancia `pgcb-mlflow` en *Running* con su *Public IPv4*.
2. **Terminal SSH**: el prompt `ec2-user@ip-…`, el comando `curl -s ifconfig.me` (muestra la IP pública) y el
   final de la salida de `python -m src.experimentos` ("Mejor en validación: …").
3. **MLflow → Experiments → `pgcb-pronostico-demanda`**: la tabla de corridas con las columnas
   `val_mape` y `prueba_mape` (botón *Columns* para agregarlas), ordenada por `val_mape`.
4. **Comparación**: seleccione las corridas `base-t24`, `ridge-rezagos_festivos-relativo`,
   `random_forest-basico_festivos-nivel` y `lightgbm-basico_festivos-relativo` → **Compare**.
5. **Detalle de la corrida `FINAL-…`**: parámetros, métricas y artefactos (`graficos/semana_prueba.png`).
6. **Model registry** (menú *Models*): `pgcb-demanda-24h`, versión 1 (y los modelos que registren los demás).

## 6. Al terminar

**EC2 → Instances → Instance state → Stop instance.** No use *Terminate*: la entrega exige dejarla
detenida. Al volver a encenderla, la IP cambia. Para retomar:

```bash
ssh -i ~/Documents/Despliegue/Despliegue.pem ec2-user@<NUEVA-IP>
cd ~/pgcb-demanda && source .venv/bin/activate
sudo swapon /swapfile
tmux new -s mlflow   # y repetir el comando de mlflow server del paso 3, con la NUEVA IP en --cors-allowed-origins
```

Los experimentos no se pierden: quedan en `mlflow.db` y `mlartifacts/` dentro de la máquina.
