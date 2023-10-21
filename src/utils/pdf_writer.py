import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

from fpdf import FPDF

"""
A class to generate PDF reports based on Mean Squared Error (MSE) values.
"""
class PDFWriter:

    def generate_box_and_whisker_plot(mse_values, save_path):
        """
        Generate a box and whisker plot for the given MSE values.

        Parameters:
        - mse_values (dict): Dictionary containing MSE values.
        - save_path (str): Path to save the generated plot.

        Returns:
        - str: Path to the saved plot image.
        """
        plt.figure(figsize=(10, 6))
        
        # Extract MSE values from the dictionary
        mse_list = list(mse_values.values())
        
        # Boxplot with custom colors for quartiles
        boxprops = dict(linestyle='-', linewidth=1, color='black')
        medianprops = dict(linestyle='-', linewidth=1, color='red')
        whiskerprops = dict(linestyle='-', linewidth=1, color='blue')
        capprops = dict(linestyle='-', linewidth=1, color='green')
        
        bp = plt.boxplot(mse_list, boxprops=boxprops, medianprops=medianprops, 
                        whiskerprops=whiskerprops, capprops=capprops)
        
        # Add grid lines to y-axis
        plt.grid(axis='y')
        
        # Adjust y-axis to have more numbers on the scale and format them to have 6 decimal places
        plt.yticks([tick for tick in plt.yticks()[0]], ['{:.6f}'.format(tick) for tick in plt.yticks()[0]])
        
        # Compute statistics
        min_val = np.min(mse_list)
        q1_val = np.percentile(mse_list, 25)
        median_val = np.median(mse_list)
        q3_val = np.percentile(mse_list, 75)
        max_val = np.max(mse_list)
        
        # Correctly identify the lines for min and max
        min_line = bp['whiskers'][0].get_ydata()[1]  # The y-coordinate for min
        max_line = bp['whiskers'][1].get_ydata()[1]  # The y-coordinate for max

        # Correctly identify the y-coordinates for Q1 and Q3
        q1_y = bp['boxes'][0].get_ydata()[0]  # The y-coordinate for Q1
        q3_y = bp['boxes'][0].get_ydata()[2]  # The y-coordinate for Q3

        # Annotate Q1 and Q3 directly using their values and positions
        plt.text(1.05, q1_y, '{:.6f}'.format(q1_val), va='center', ha='center')
        plt.text(1.05, q3_y, '{:.6f}'.format(q3_val), va='center', ha='center')

        # Annotate min and max using their lines
        plt.text(1.05, min_line, '{:.6f}'.format(min_val), va='center', ha='center')
        plt.text(1.05, max_line, '{:.6f}'.format(max_val), va='center', ha='center')

        # Annotate median
        plt.text(1.05, bp['medians'][0].get_ydata()[0], '{:.6f}'.format(median_val), va='center', ha='center')
        
        plt.title("Box and Whisker Plot for MSE Values")
        plt.ylabel("MSE Value")
        plt.xlabel("Files")
        
        image_path = os.path.join(save_path, "mse_box_plot.png")
        plt.savefig(image_path)
        plt.close()
        
        return image_path



    def generate_histogram_plot(mse_values, save_path):
        """
        Generate a histogram for the given MSE values.

        Parameters:
        - mse_values (dict): Dictionary containing MSE values.
        - save_path (str): Path to save the generated histogram.

        Returns:
        - str: Path to the saved histogram image.
        """
        plt.figure(figsize=(10, 6))
        
        # Extract MSE values from the dictionary
        mse_list = list(mse_values.values())
        
        # Plot histogram
        plt.hist(mse_list, bins=30, color='skyblue', edgecolor='black')
        
        plt.title("Histogram of MSE Values")
        plt.ylabel("Frequency")
        plt.xlabel("MSE Value")
        
        image_path = os.path.join(save_path, "mse_histogram.png")
        plt.savefig(image_path)
        plt.close()
        
        return image_path


    def generate_pdf_report(checkpoint_path, timestamp, mse_values, final_save_path, images):
        """
        Generate a PDF report based on the given MSE values and images.

        Parameters:
        - checkpoint_path (str): Path to the checkpoint file.
        - timestamp (str): Date and time of the report generation.
        - mse_values (dict): Dictionary containing MSE values.
        - final_save_path (str): Path to save the final PDF report.
        - images (list): List of image paths to be included in the report.

        Note:
        - The generated report will include a box and whisker plot, histogram, and a table of MSE values.
        - The report will also include the top images based on MSE values.
        """
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "Evaluation Report", 0, 1, 'C')
        
        # Add checkpoint name and path
        pdf.set_font("Arial", size=12)
        pdf.ln(10)
        pdf.cell(200, 10, f"Checkpoint: {os.path.basename(checkpoint_path)}", 0, 1)
        pdf.cell(200, 10, f"Path: {checkpoint_path}", 0, 1)

        # Add datetime
        pdf.cell(200, 10, f"Date and Time: {timestamp}", 0, 1)
        
        # Add box and whisker plot for MSE values
        mse_plot_path = PDFWriter.generate_box_and_whisker_plot(mse_values, final_save_path)
        pdf.image(mse_plot_path, x = 10, y = pdf.get_y(), w = 190)  

        # Add histogram plot for MSE values on a new page
        pdf.add_page()
        mse_histogram_path = PDFWriter.generate_histogram_plot(mse_values, final_save_path)
        pdf.image(mse_histogram_path, x = 10, y = pdf.get_y(), w = 190)  

        # Start table on a new page
        pdf.add_page()
        
        # Set column widths based on specified percentages
        table_width        = 190  
        filename_col_width = 0.75 * table_width
        mse_col_width      = 0.25 * table_width
        
        # Add table with all MSE values sorted by value
        sorted_mse = sorted(mse_values.items(), key=lambda x: x[1])
        pdf.cell(filename_col_width, 10, "File Name", 1)  # Column header
        pdf.cell(mse_col_width, 10, "MSE Value", 1)  # Column header
        pdf.ln()
        for filename, mse in sorted_mse:
            pdf.cell(filename_col_width, 10, filename, 1)
            pdf.cell(mse_col_width, 10, str(mse), 1)
            pdf.ln()

        # Add the top images to the PDF
        for rank, image_path in images:
            pdf.add_page()
            if rank < 3:
                heading = f"{rank + 1} Lowest MSE Value - {os.path.basename(image_path).replace('_predicted.png', '')}"
            else:
                heading = f"Highest MSE Value - {os.path.basename(image_path).replace('_predicted.png', '')}"
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, heading, 0, 1, 'C')
            pdf.image(image_path, x=10, y=pdf.get_y(), w=190)

        # Delete the temp folder used to store images
        shutil.rmtree("temp")
        
        # Save PDF to the evaluation folder
        pdf_output_path = os.path.join(final_save_path, "evaluation_report.pdf")
        pdf.output(pdf_output_path)
        print(f"PDF report saved to {pdf_output_path}")