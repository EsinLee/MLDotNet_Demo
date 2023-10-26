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

namespace MLDemo
{
    /// <summary>
    /// MainWindow.xaml 的互動邏輯
    /// </summary>
    public partial class MainWindow : Window
    {
        // The classification categories of existing image classification models
        string[] classNames = new string[] { "cat","dog" };

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

                var prediction = await Task.Run(() =>
                {
                    var context = new MLContext();
                    var model = context.Model.Load("./MLModel/cat_and_dog-model.zip", out DataViewSchema inputSchema);
                    var predictionEngine = context.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);

                    var imageBytes = File.ReadAllBytes(target_image_path);
                    return predictionEngine.Predict(new ImageData
                    {
                        ImageSource = imageBytes
                    });
                });

                Result.Text = $"Prediction salary is {prediction.PredictedLabel} -> {prediction.Score[Array.IndexOf(classNames, prediction.PredictedLabel)]}";
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
    }
}
