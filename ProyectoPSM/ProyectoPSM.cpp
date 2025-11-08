#include "ProyectoPSM.h"
#include "ui_ProyectoPSM.h"


ProyectoPSM::ProyectoPSM(QWidget* parent)
    : QMainWindow(parent), ui(new Ui::ProyectoPSM)
{
    ui->setupUi(this);

	// Inicializar combos
    inicializarCombos();
    temporizador = new QTimer(this);
    Recording = false;

    // Conectar señales y slots
    connect(ui->btnStart, SIGNAL(clicked()), this, SLOT(iniciarDetenerGrabacion()));
    connect(&camara, SIGNAL(NewImageSignal()), this, SLOT(GetImage()));
    connect(temporizador, SIGNAL(timeout()), this, SLOT(actualizarFrame()));
    connect(ui->btnCapture, SIGNAL(clicked()), this, SLOT(capturarImagen()));
    

    // Crear carpeta dataset si no existe
    QDir dir;
    if (!dir.exists("dataset")) dir.mkdir("dataset");
}

ProyectoPSM::~ProyectoPSM()
{
    camara.~CVideoAcquisition();
    delete ui;
}

void ProyectoPSM::inicializarCombos()
{
    QStringList codigos = { "01","02","03" };
    ui->comboCodigo->addItems(codigos);

    QStringList azimuth = { "000","045","090","135","180","225","270","315" };
    ui->comboAzimuth->addItems(azimuth);

	QStringList elevacion = { "000","030","060","090" };
    ui->comboElev->addItems(elevacion);

	QStringList secuencias = { "001","002","003","004","005" };
    ui->comboSeq->addItems(secuencias);
}

void ProyectoPSM::iniciarDetenerGrabacion()
{
    if (!camara.CameraOK) {
        QMessageBox::warning(this, "Error", "No se pudo abrir la cámara.");
        return;
    }

    if (!Recording) {
        ui->btnStart->setStyleSheet("background-color: #E05334");
        ui->btnStart->setText("Parar");
        camara.SetCameraAutoExposure();
		camara.StartStopCapture(true);
        temporizador->start(30);            // 30 ms ~ 33 fps
		Recording = true;
    }
    else {
        ui->btnStart->setStyleSheet("background-color: white");
        ui->btnStart->setText("Iniciar");
        camara.StartStopCapture(false);
        temporizador->stop();
		Recording = false;
    }
}

void ProyectoPSM::actualizarFrame()
{
	frameActual = camara.GetImage();

    if (frameActual.empty()) return;

    // Convertimos a QImage para mostrar
    QImage imagen = matToQImage(frameActual);
    ui->labelPreview->setPixmap(QPixmap::fromImage(imagen).scaled(
        ui->labelPreview->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void ProyectoPSM::capturarImagen()
{
    if (frameActual.empty()) {
        QMessageBox::warning(this, "Captura", "No hay imagen disponible.");
        return;
    }

    QString nombre = generarNombreArchivo();
    imwrite(nombre.toStdString(), frameActual);
    QMessageBox::information(this, "Imagen guardada", nombre);
}

QString ProyectoPSM::generarNombreArchivo()
{
    QString codigo = ui->comboCodigo->currentText();
    QString az = ui->comboAzimuth->currentText();
    QString el = ui->comboElev->currentText();
    QString seq = ui->comboSeq->currentText();
    return QString("dataset/%1_%2_%3_%4.png").arg(codigo, az, el, seq);
}

QImage ProyectoPSM::matToQImage(const Mat& mat)
{
    // Si la imagen es a color
    if (mat.type() == CV_8UC3) {
        Mat rgb;
        cvtColor(mat, rgb, COLOR_BGR2RGB);
        return QImage(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888).copy();
    }
    // Si es a escala de grises
    else if (mat.type() == CV_8UC1) {
        return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8).copy();
    }
    else {
        return QImage();
    }
}
