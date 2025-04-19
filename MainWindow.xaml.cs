using System;
using System.IO;
using System.Net.Sockets;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media.Imaging;
using System.Windows.Media;
using Microsoft.Kinect;

namespace Microsoft.Samples.Kinect.DepthBasics
{
    public partial class MainWindow : Window
    {
        private KinectSensor sensor;

        // Depth
        private DepthImagePixel[] depthPixels;
        private byte[] depthColorPixels;
        private WriteableBitmap depthBitmap;

        // Color
        private byte[] rgbPixels;
        private WriteableBitmap rgbBitmap;

        // Heatmap
        private byte[] heatmapColorPixels; // To store the heatmap pixel data
        private WriteableBitmap heatmapBitmap;

        // Networking
        private TcpClient depthClient;
        private NetworkStream depthStream;
        private TcpClient colorClient;
        private NetworkStream colorStream;

        public MainWindow()
        {
            InitializeComponent();
        }

        private void WindowLoaded(object sender, RoutedEventArgs e)
        {
            // Find first connected Kinect
            foreach (var potential in KinectSensor.KinectSensors)
                if (potential.Status == KinectStatus.Connected)
                {
                    sensor = potential;
                    break;
                }

            if (sensor == null)
            {
                statusBarText.Text = Properties.Resources.NoKinectReady;
                return;
            }

            // DEPTH stream
            sensor.DepthStream.Enable(DepthImageFormat.Resolution640x480Fps30);
            depthPixels = new DepthImagePixel[sensor.DepthStream.FramePixelDataLength];
            depthColorPixels = new byte[sensor.DepthStream.FramePixelDataLength * sizeof(int)];
            depthBitmap = new WriteableBitmap(
                sensor.DepthStream.FrameWidth,
                sensor.DepthStream.FrameHeight,
                96, 96,
                PixelFormats.Bgr32, null);
            Image.Source = depthBitmap;
            sensor.DepthFrameReady += SensorDepthFrameReady;

            // COLOR stream
            sensor.ColorStream.Enable(ColorImageFormat.RgbResolution640x480Fps30);
            rgbPixels = new byte[sensor.ColorStream.FramePixelDataLength]; // 640*480*4
            rgbBitmap = new WriteableBitmap(
                sensor.ColorStream.FrameWidth,
                sensor.ColorStream.FrameHeight,
                96, 96,
                PixelFormats.Bgr32, null);
            RgbImage.Source = rgbBitmap;
            sensor.ColorFrameReady += SensorColorFrameReady;

            // Heatmap
            heatmapColorPixels = new byte[sensor.DepthStream.FramePixelDataLength * 4]; // RGBA format
            heatmapBitmap = new WriteableBitmap(
                sensor.DepthStream.FrameWidth,
                sensor.DepthStream.FrameHeight,
                96, 96,
                PixelFormats.Bgr32, null);
            HeatmapImage.Source = heatmapBitmap;

            // Open TCP
            colorClient = new TcpClient("127.0.0.1", 9999);
            colorStream = colorClient.GetStream();
            depthClient = new TcpClient("127.0.0.1", 9998);
            depthStream = depthClient.GetStream();

            // Start
            try { sensor.Start(); }
            catch (IOException) { sensor = null; }
        }

        private void WindowClosing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            if (sensor != null)
                sensor.Stop();
        }

        private void SensorDepthFrameReady(object sender, DepthImageFrameReadyEventArgs e)
        {
            using (var frame = e.OpenDepthImageFrame())
            {
                if (frame == null) return;

                frame.CopyDepthImagePixelDataTo(depthPixels);
                int idx = 0;
                for (int i = 0; i < depthPixels.Length; i++)
                {
                    short d = depthPixels[i].Depth;
                    byte r = 0, g = 0, b = 0;
                    if (d == 0) { }
                    else if (d < 1000) { r = 255; }
                    else if (d < 2000) { r = 255; g = 128; }
                    else if (d < 3000) { r = 255; g = 255; }
                    else if (d < 4000) { g = 255; }
                    else { b = 255; }
                    depthColorPixels[idx++] = b;
                    depthColorPixels[idx++] = g;
                    depthColorPixels[idx++] = r;
                    idx++; // skip alpha
                }

                ushort[] raw = new ushort[depthPixels.Length];
                for (int i = 0; i < depthPixels.Length; i++)
                    raw[i] = (ushort)depthPixels[i].Depth;
                byte[] depthBytes = new byte[raw.Length * sizeof(ushort)];
                Buffer.BlockCopy(raw, 0, depthBytes, 0, depthBytes.Length);
                SendWithBigEndianLength(depthStream, depthBytes);

                // Update depth bitmap
                depthBitmap.WritePixels(
                    new Int32Rect(0, 0, depthBitmap.PixelWidth, depthBitmap.PixelHeight),
                    depthColorPixels,
                    depthBitmap.PixelWidth * sizeof(int),
                    0);

                // Update heatmap
                UpdateHeatmap();
            }
        }

        private void SensorColorFrameReady(object sender, ColorImageFrameReadyEventArgs e)
        {
            using (var frame = e.OpenColorImageFrame())
            {
                if (frame == null) return;

                // Copy BGRX (4 bytes per pixel)
                frame.CopyPixelDataTo(rgbPixels);
                rgbBitmap.WritePixels(
                    new Int32Rect(0, 0, rgbBitmap.PixelWidth, rgbBitmap.PixelHeight),
                    rgbPixels,
                    rgbBitmap.PixelWidth * 4,
                    0);

                // JPEG encode via WPF
                byte[] jpeg;
                var encoder = new JpegBitmapEncoder();
                encoder.Frames.Add(BitmapFrame.Create(rgbBitmap));
                using (var ms = new MemoryStream())
                {
                    encoder.Save(ms);
                    jpeg = ms.ToArray();
                }
                SendWithBigEndianLength(colorStream, jpeg);
            }
        }

        private void UpdateHeatmap()
        {
            int idx = 0;
            for (int i = 0; i < depthPixels.Length; i++)
            {
                short d = depthPixels[i].Depth;
                byte r = 0, g = 0, b = 0;

                // Heatmap logic based on distance (depth value)
                if (d == 0) { }
                else if (d < 1000) { r = 255; }
                else if (d < 2000) { r = 255; g = 128; }
                else if (d < 3000) { r = 255; g = 255; }
                else if (d < 4000) { g = 255; }
                else { b = 255; }

                // Store heatmap colors (RGBA format)
                heatmapColorPixels[idx++] = b;
                heatmapColorPixels[idx++] = g;
                heatmapColorPixels[idx++] = r;
                heatmapColorPixels[idx++] = 255; // alpha (fully opaque)
            }

            // Update heatmap bitmap
            heatmapBitmap.WritePixels(
                new Int32Rect(0, 0, heatmapBitmap.PixelWidth, heatmapBitmap.PixelHeight),
                heatmapColorPixels,
                heatmapBitmap.PixelWidth * 4,
                0);
        }

        private void SendWithBigEndianLength(NetworkStream stream, byte[] data)
        {
            byte[] len = BitConverter.GetBytes(data.Length);
            if (BitConverter.IsLittleEndian)
                Array.Reverse(len);
            stream.Write(len, 0, 4);
            stream.Write(data, 0, data.Length);
        }

        private void TiltSlider_MouseReleased(object sender, MouseButtonEventArgs e)
        {
            if (sensor != null && sensor.IsRunning)
            {
                try
                {
                    sensor.ElevationAngle = (int)TiltSlider.Value;
                    statusBarText.Text = $"Set elevation angle to {(int)TiltSlider.Value}°";
                }
                catch { statusBarText.Text = "Failed to set elevation angle."; }
            }
        }

        private void TiltSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            TiltValueLabel.Content = $"{(int)e.NewValue}°";
        }
    }
}
