const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs-extra');
const ffmpeg = require('fluent-ffmpeg');
const tf = require('@tensorflow/tfjs-node');
const blazeface = require('@tensorflow-models/blazeface');
const { createCanvas, loadImage } = require('canvas');
const cors = require('cors');

const app = express();
const PORT = 3000;

// Configuração do multer para upload
const upload = multer({ dest: 'uploads/' });

// Classe para processamento de vídeo
class VideoFaceRecognition {
  constructor(videoPath, outputDir) {
    this.videoPath = videoPath;
    this.outputDir = outputDir;
  }

  async extractFrames(fps = 24) {
    const framesDir = path.join(this.outputDir, 'frames');
    await fs.ensureDir(framesDir);

    return new Promise((resolve, reject) => {
      ffmpeg(this.videoPath)
        .output(path.join(framesDir, 'frame-%04d.jpg'))
        .outputOptions([`-vf fps=${fps}`])
        .on('end', () => resolve(framesDir))
        .on('error', reject)
        .run();
    });
  }

  async processFrames(framesDir) {
    const model = await blazeface.load();
    const frames = await fs.readdir(framesDir);
    const processedFramesDir = path.join(this.outputDir, 'processed_frames');
    await fs.ensureDir(processedFramesDir);

    for (const frame of frames) {
      const framePath = path.join(framesDir, frame);
      const img = await loadImage(framePath);

      const canvas = createCanvas(img.width, img.height);
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, img.width, img.height);

      const inputTensor = tf.browser.fromPixels(canvas);
      const predictions = await model.estimateFaces(inputTensor, false);

      if (predictions.length > 0) {
        predictions.forEach((face) => {
          if (face.topLeft && face.bottomRight) {
            const [x1, y1] = face.topLeft;
            const [x2, y2] = face.bottomRight;
            const width = x2 - x1;
            const height = y2 - y1;

            // Desenhar os cantos do rosto
            const cornerLength = Math.min(width, height) * 0.2;
            ctx.strokeStyle = 'yellow';
            ctx.lineWidth = 5;

            // Top-left corner
            ctx.beginPath();
            ctx.moveTo(x1, y1 + cornerLength);
            ctx.lineTo(x1, y1);
            ctx.lineTo(x1 + cornerLength, y1);
            ctx.stroke();

            // Top-right corner
            ctx.beginPath();
            ctx.moveTo(x2 - cornerLength, y1);
            ctx.lineTo(x2, y1);
            ctx.lineTo(x2, y1 + cornerLength);
            ctx.stroke();

            // Bottom-right corner
            ctx.beginPath();
            ctx.moveTo(x2, y2 - cornerLength);
            ctx.lineTo(x2, y2);
            ctx.lineTo(x2 - cornerLength, y2);
            ctx.stroke();

            // Bottom-left corner
            ctx.beginPath();
            ctx.moveTo(x1 + cornerLength, y2);
            ctx.lineTo(x1, y2);
            ctx.lineTo(x1, y2 - cornerLength);
            ctx.stroke();
          }
        });

        const outputFramePath = path.join(processedFramesDir, frame);
        const buffer = canvas.toBuffer('image/jpeg');
        fs.writeFileSync(outputFramePath, buffer);
      }

      inputTensor.dispose();
    }

    return processedFramesDir;
  }

  async remountVideo(framesDir, outputVideoPath, fps = 24) {
    const processedFrames = await fs.readdir(framesDir);

    if (processedFrames.length === 0) {
      throw new Error('Nenhum frame processado encontrado para remontar o vídeo.');
    }

    return new Promise((resolve, reject) => {
      ffmpeg(path.join(framesDir, 'frame-%04d.jpg'))
        .inputOptions([`-framerate ${fps}`])
        .output(outputVideoPath)
        .outputOptions(['-pix_fmt yuv420p']) // Compatibilidade com players
        .on('end', () => resolve(outputVideoPath))
        .on('error', reject)
        .run();
    });
  }

  async processVideo(fps = 24) {
    const framesDir = await this.extractFrames(fps);
    console.log('Frames extraídos para:', framesDir);

    const processedFramesDir = await this.processFrames(framesDir);
    console.log('Frames processados para:', processedFramesDir);

    const outputVideoPath = path.join(this.outputDir, 'processed_video.mp4');
    await this.remountVideo(processedFramesDir, outputVideoPath, fps);

    console.log(`Vídeo processado salvo em: ${outputVideoPath}`);
    return outputVideoPath;
  }
}

// Middleware CORS
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Endpoint para upload e processamento de vídeo
app.post('/process-video', upload.single('video'), async (req, res) => {
  try {
    const videoFile = req.file;
    if (!videoFile) {
      return res.status(400).json({ error: 'Nenhum vídeo enviado.' });
    }

    const outputDir = path.join(__dirname, 'output', videoFile.filename);
    await fs.ensureDir(outputDir);

    const videoProcessor = new VideoFaceRecognition(videoFile.path, outputDir);

    console.log('Iniciando processamento do vídeo...');
    const processedVideoPath = await videoProcessor.processVideo(24);

    console.log('Processamento concluído:', processedVideoPath);

    // Enviar o vídeo processado para o cliente
    console.log('Tentando enviar o vídeo processado:', processedVideoPath);
    res.download(processedVideoPath, 'processed_video.mp4', async (err) => {
      if (err) {
        console.error('Erro ao enviar vídeo:', err);
        return res.status(500).json({ error: 'Erro ao enviar o vídeo processado.' });
      }

      console.log('Vídeo enviado com sucesso!');
      // Limpeza de arquivos temporários após envio
      await fs.remove(videoFile.path);
      await fs.remove(outputDir);
    });
  } catch (err) {
    console.error('Erro durante o processamento:', err);
    res.status(500).json({ error: 'Erro no processamento do vídeo.' });
  }
});

// Iniciar o servidor
app.listen(PORT, () => {
  console.log(`Servidor rodando em http://localhost:${PORT}`);
});
