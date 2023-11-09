using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

using System.IO;
using Microsoft.ML;

using MLDemo;
using MLDemo.Models;
using Microsoft.ML.Data;
using Tensorflow;
using Path = System.IO.Path;
using System.Net;
using System.Diagnostics;
using System.Windows.Threading;
using System.Runtime.Remoting.Metadata.W3cXsd2001;

namespace MLDemo
{
    /// <summary>
    /// MainWindow.xaml 的互動邏輯
    /// </summary>
    public partial class MainWindow : Window
    {
        // The classification categories of existing image classification models
        string[] classNames = new string[] { "cat","dog" };
        // 這是您的類的字段
        private System.Windows.Threading.DispatcherTimer timer;
        private System.Diagnostics.Stopwatch stopwatch = new System.Diagnostics.Stopwatch();

        public MainWindow()
        {
            InitializeComponent();

            BitmapImage image = new BitmapImage(new Uri("Images/cat_01.jpg", UriKind.RelativeOrAbsolute));
            //Img_main.Source = new BitmapImage(new Uri("./Images/cat_01.jpg", UriKind.RelativeOrAbsolute));
            Img_main.Source = image;

            if (!image.UriSource.IsAbsoluteUri)
            {
                string fullPath = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, image.UriSource.ToString());
                //Console.WriteLine(fullPath);
                YearsOfExperience.Text = fullPath;
            }
            else
            {
                //Console.WriteLine(image.UriSource.ToString());
                YearsOfExperience.Text = image.UriSource.ToString();
            }
        }

        // Prediction of image categories
        private async void BtnEvent_Click_Predict(object sender, RoutedEventArgs e)
        {
            /*var context = new MLContext();
            var model = context.Model.Load("./MLModel/salary-model.zip", out DataViewSchema inputSchema);
            var predictionEngine = context.Model.CreatePredictionEngine<SalaryData, SalaryPrediction>(model, inputSchema: inputSchema);
            var prediction = predictionEngine.Predict(new SalaryData
            {
                YearsExperience = float.Parse(YearsOfExperience.Text)
            });
            Result.Text = $"Prediction salary is {prediction.PredictedSalary.ToString("c")}";*/

            string target_image_path = YearsOfExperience.Text == "" ? @"D:\repos\MLDotNetSampleProject\MLDotNetSampleProject\assets\images\cat\cat_1.jpg" : YearsOfExperience.Text;
            if (File.Exists(target_image_path))
            {
                BitmapImage image = new BitmapImage(new Uri(target_image_path, UriKind.Absolute));
                Img_main.Source = image;
                Loading_Mask.Visibility = Visibility.Visible; // show loading mask

                // 初始化計時器UI
                txb_training_time.Text = "00:00:00";
                stopwatch.Reset(); // 重置stopwatch
                stopwatch.Start(); // 啟動stopwatch
                // 設定和開始DispatcherTimer
                timer = new DispatcherTimer();
                timer.Interval = TimeSpan.FromSeconds(1);  // 每秒更新一次
                timer.Tick += Training_Timer_Tick;
                timer.Start();

                var prediction = await Task.Run(() =>
                {
                    var context = new MLContext();
                    var model = context.Model.Load("./MLModel/cat_and_dog-model.zip", out DataViewSchema inputSchema);
                    //var model = context.Model.Load("./MLModel/retrainedModel.zip", out DataViewSchema inputSchema);
                    var predictionEngine = context.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);

                    var imageBytes = File.ReadAllBytes(target_image_path);
                    return predictionEngine.Predict(new ImageData
                    {
                        ImageSource = imageBytes
                    });
                });

                timer.Stop(); // 停止DispatcherTimer
                stopwatch.Stop(); // 停止stopwatch

                //Result.Text = $"Prediction result : {prediction.PredictedLabel}";
                Result.Text = $"Prediction result : {prediction.PredictedLabel} -> {prediction.Score[Array.IndexOf(classNames, prediction.PredictedLabel)]}";
                Loading_Mask.Visibility = Visibility.Hidden; // hide loading mask
            }
            else
            {
                Result.Text = "Error - Image not exist.";
            }
        }

        // Drop in local image and show up
        private void ImageDrop_Display(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                // Get the dropped files
                string[] files = (string[])e.Data.GetData(DataFormats.FileDrop);

                // Just take the first one if there are multiple
                string filePath = files[0];

                // Check if it's an image by examining the file extension
                string ext = Path.GetExtension(filePath).ToLower();
                if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".gif")
                {
                    // Set the image source
                    Img_main.Source = new BitmapImage(new Uri(filePath));
                    YearsOfExperience.Text = filePath;
                }
                else
                {
                    MessageBox.Show("Not a valid image file!");
                }
                Result.Text = "";
            }
        }

        #region Train Model
        private async void Training_Model(object sender, RoutedEventArgs e)
        {
            Loading_Mask.Visibility = Visibility.Visible; // show loading mask

            // 初始化計時器UI
            txb_training_time.Text = "00:00:00";
            // 初始化計時器
            stopwatch.Reset(); // 重置stopwatch
            stopwatch.Start();  // 開始計時
            // 設定和開始DispatcherTimer
            timer = new DispatcherTimer();
            timer.Interval = TimeSpan.FromSeconds(1);  // 每秒更新一次
            timer.Tick += Training_Timer_Tick;
            timer.Start();

            var training_task = Task.Run(() =>
            {
                // 1. 初始化MLContext
                MLContext mlContext = new MLContext();

                // 2. 從"asset"目錄加載圖像數據
                //var images = LoadImagesFromDirectory(directory: "assets", useFolderNameAsLabel: true);
                var images = LoadImagesFromDirectory(directory: "extra_training_data", useFolderNameAsLabel: true);
                IDataView trainData = mlContext.Data.LoadFromEnumerable(images);

                // 3. 使用RetrainPipeline函數
                // Data process configuration with pipeline data transformations
                var pipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: @"Label", inputColumnName: @"Label")
                                        .Append(mlContext.MulticlassClassification.Trainers.ImageClassification(labelColumnName: @"Label", scoreColumnName: @"Score", featureColumnName: @"ImageSource"))
                                        .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: @"PredictedLabel", inputColumnName: @"PredictedLabel"));
                var retrainedModel = pipeline.Fit(trainData);

                // 7. 保存重新訓練的模型
                mlContext.Model.Save(retrainedModel, trainData.Schema, "MLModel/retrainedModel.zip");
            });
            // Wait for the training process to complete
            await training_task;
            
            timer.Stop(); // 停止DispatcherTimer
            stopwatch.Stop(); // 停止計時
            TimeSpan ts = stopwatch.Elapsed;
            // 格式化和顯示時間
            string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}",
                ts.Hours, ts.Minutes, ts.Seconds);
            MessageBox.Show($"訓練完成! 花費時間: {elapsedTime}", "訓練時間");

            Loading_Mask.Visibility = Visibility.Hidden; // hide loading mask
        }
        private void Training_Timer_Tick(object sender, EventArgs e)
        {
            TimeSpan ts = stopwatch.Elapsed;
            string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}",
                ts.Hours, ts.Minutes, ts.Seconds);
            // 更新UI顯示經過的時間
            txb_training_time.Text = elapsedTime;
        }
        public static IEnumerable<ImageTrainData> LoadImagesFromDirectory(string directory, bool useFolderNameAsLabel = true)
        {
            var imagesPath = Directory.GetFiles(directory, "*", searchOption: SearchOption.AllDirectories)
                .Where(s => s.EndsWith(".jpeg") || s.EndsWith(".png") || s.EndsWith(".jpg") || s.EndsWith(".tif") || s.EndsWith(".tiff"));

            return imagesPath.Select(imagePath => new ImageTrainData
            {
                //ImageSource = imagePath,

                ImageSource = File.ReadAllBytes(imagePath),
                Label = useFolderNameAsLabel ? Directory.GetParent(imagePath).Name : Path.GetFileName(imagePath)
            });
        }
        #endregion
    }
}
