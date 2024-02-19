from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# nltk.download('punkt') 
# nltk.download('stopwords') 
qa_pairs = [
    {
        "Answer": "The primary use of the Matplotlib library in Python is for drawing charts and general visualization. It is widely used for creating various types of visualizations such as line graphs, bar charts, scatter plots, histograms, etc., to represent data in a graphical format.",
        "Question": "1. What is the primary use of the Matplotlib library in Python?"
    },
    {
        "Answer": "To install the Matplotlib library using pip, you can use the following command in your terminal or command prompt:\n\n```\npython3 -m pip install matplotlib\n```\n\nThis command will download and install the Matplotlib library along with any necessary dependencies.",
        "Question": "2. How can you install the Matplotlib library using pip?"
    },
    {
        "Answer": "Open-source libraries and applications play a significant role in today's computing world due to several reasons:\n\n1. **Accessibility**: Open-source software is freely available to everyone, which promotes accessibility and allows users to utilize high-quality programs without financial barriers.\n\n2. **Transparency**: The source code of open-source software is openly available for inspection, modification, and enhancement. This transparency fosters trust, security, and collaboration among developers and users.\n\n3. **Customization**: Users have the freedom to modify open-source software to suit their specific needs, enabling customization and flexibility that may not be possible with proprietary software.\n\n4. **Community Collaboration**: Open-source projects often have vibrant communities of developers and users who contribute to the improvement and evolution of the software. This collaborative environment leads to rapid innovation and continuous development.\n\n5. **Education and Learning**: Open-source software provides an excellent platform for learning and skill development. Users can study the source code, experiment, and gain practical experience in software development.\n\n6. **Innovation**: Open-source libraries and applications have been instrumental in driving innovation across various domains. They serve as building blocks for new technologies and solutions, encouraging creativity and advancement in the tech industry.\n\nOverall, open-source libraries and applications have democratized access to technology, promoted collaboration, and accelerated progress in the computing world.",
        "Question": "3. What is the significance of open-source libraries and applications in today's computing world?"
    },
    {
        "Answer": "Some of the most notable open-source programs mentioned in the provided context are:\n\n1. The Linux operating system\n2. GIMP (GNU Image Manipulation Program)\n3. The Python language\n4. Firefox web browser\n5. LibreOffice\n6. VLC video player",
        "Question": "4. Name some of the most notable open-source programs."
    },
    {
        "Answer": "The `zip()` function in Python is used to combine multiple iterables into a single iterable of tuples. It takes iterables as arguments and returns an iterator that generates tuples containing elements from each iterable. In the context of the lecture, the `zip()` function was used to sort and extract data from two lists simultaneously. Specifically, it was used to combine state abbreviations and their corresponding 8th-grade reading scores into tuples, which were then sorted based on the reading scores.",
        "Question": "5. What is the zip() function in Python and how is it used in the context of the lecture?"
    },
    {
        "Answer": "To create a simple plot using Matplotlib and numpy, you can follow these steps:\n\n1. Import the necessary libraries:\n```python\nimport matplotlib.pyplot as plt\nimport numpy as np\n```\n\n2. Create your data using numpy:\n```python\nx = np.linspace(0, 2, 100)  # numpy array with 100 numbers evenly spaced from 0 to 2\n```\n\n3. Plot the data using Matplotlib:\n```python\nplt.plot(x, x, label='linear')  # Plot some data on the (implicit) axes.\nplt.plot(x, x ** 2, label='quadratic')\nplt.plot(x, x ** 3, label='cubic')\nplt.xlabel('x label')\nplt.ylabel('y label')\nplt.title(\"Simple Plot\")\nplt.legend()\nplt.show()\n```\n\nThis code will generate a simple plot with three lines representing linear, quadratic, and cubic functions. The x-axis will be labeled 'x label' and span from 0.00 to 2.00, while the y-axis will be labeled 'y label' and span from 0 to 8. The title of the graph will be \"Simple Plot\" with a legend showing the function types.",
        "Question": "6. How can you create a simple plot using Matplotlib and numpy?"
    },
    {
        "Answer": "Using numpy arrays is generally faster and more efficient compared to using regular Python lists for numerical computations. Numpy arrays are optimized for numerical operations and are implemented in C, which makes them faster for large datasets. Numpy arrays also provide a wide range of mathematical functions and operations that can be applied directly to the entire array, making it more convenient for numerical computations.\n\nIn terms of functionality, numpy arrays offer a variety of mathematical functions, broadcasting capabilities, and vectorized operations that are not directly available in regular Python lists. Numpy arrays also support multidimensional arrays and have a rich set of functions for linear algebra, Fourier transforms, random number generation, etc.\n\nOverall, numpy arrays are preferred for numerical computations and data manipulation tasks due to their speed, efficiency, and rich functionality compared to regular Python lists.",
        "Question": "7. How does using numpy arrays compare to using regular Python lists in terms of speed and functionality?"
    },
    {
        "Answer": "To visualize real data sets using Matplotlib, you can follow these general steps:\n\n1. Import the necessary libraries: Import Matplotlib and any other libraries needed for data manipulation, such as NumPy for numerical operations.\n\n2. Load your data: Read the real data set you want to visualize. This can be done using various methods depending on the format of your data (e.g., CSV, Excel, SQL databases).\n\n3. Prepare your data: Extract the relevant information from your dataset that you want to visualize. This may involve cleaning, filtering, and transforming the data.\n\n4. Choose a plot type: Decide on the type of visualization that best represents your data. Matplotlib offers a wide range of plot types, including line plots, bar charts, scatter plots, histograms, and more.\n\n5. Create the plot: Use Matplotlib functions to create the chosen plot type and customize it as needed. Set labels for axes, add a title, and include legends if necessary.\n\n6. Display the plot: Show the plot using `plt.show()` to visualize it interactively. You can also save the plot as an image file using `plt.savefig()`.\n\n7. Explore and analyze: Once the plot is displayed, analyze the visualized data to draw insights and conclusions from the visualization.\n\nIn the provided context, there is an example function `eighth_grade_reading_scores_2019()` that demonstrates how to visualize average 8th-grade reading scores for U.S. states using Matplotlib. This function reads data from a CSV file, processes it, and creates a bar chart to represent the data. You can adapt this approach to visualize your own real data sets with Matplotlib.",
        "Question": "8. How can you use Matplotlib to visualize real data sets?"
    },
    {
        "Answer": "The `numpy.arange` function is used to create an array of evenly spaced values within a specified range. In the context of creating bar charts, `numpy.arange` is often used to generate the positions of the bars along the x-axis. By specifying the length of the array and the width of the bars, you can effectively position the bars at regular intervals on the chart.",
        "Question": "9. What is the purpose of the numpy.arange function in the context of creating bar charts?"
    },
    {
        "Answer": "Some alternatives to Matplotlib for data visualization include:\n\n1. **Seaborn**: Seaborn is built on top of Matplotlib and provides a high-level interface for creating attractive and informative statistical graphics. It is particularly useful for visualizing statistical relationships in data.\n\n2. **Plotly**: Plotly is an interactive, web-based plotting library that allows for creating interactive plots and dashboards. It offers a wide range of visualization options and is well-suited for creating interactive visualizations for web applications.\n\n3. **Bokeh**: Bokeh is another interactive visualization library that targets modern web browsers for presentation. It is designed for creating interactive plots, dashboards, and data applications.\n\n4. **Altair**: Altair is a declarative statistical visualization library for Python that is based on Vega and Vega-Lite. It allows for creating complex visualizations with concise and intuitive syntax.\n\n5. **ggplot**: ggplot is a Python implementation of the popular ggplot2 library in R. It follows the grammar of graphics approach and provides a high-level API for creating complex visualizations.\n\nThe choice of visualization library depends on the specific requirements of the project. Some libraries may offer more interactive features, better aesthetics, or easier syntax for certain types of visualizations. It's important to consider factors like ease of use, interactivity, customization options, and the type of visualizations needed when selecting a library for data visualization.",
        "Question": "10. What are some alternatives to Matplotlib for data visualization and why might you choose them?"
    },
    {
        "Answer": "You can save a plot programmatically in Matplotlib using the `savefig()` function. Here is an example of how you can save a plot as an image file before displaying it:\n\n```python\nimport matplotlib.pyplot as plt\n\n# Create your plot\n\nplt.savefig('my-plot.png')  # Save the plot as an image file\nplt.show()  # Display the plot\n```\n\nIn this example, the `savefig()` function is used to save the plot as an image file named 'my-plot.png' before displaying it using `plt.show()`.",
        "Question": "11. How can you save a plot programmatically in Matplotlib?"
    },
    {
        "Answer": "To create a plot with multiple lines in Matplotlib, you can use the `plt.plot()` function multiple times with different datasets. Each time you call `plt.plot()`, it will add a new line to the plot. You can also use `plt.legend()` to add a legend to differentiate between the lines. Here's a basic example:\n\n```python\nimport matplotlib.pyplot as plt\nimport numpy as np\n\nx = np.linspace(0, 2, 100)\n\nplt.plot(x, x, label='linear')  # Plot a linear line\nplt.plot(x, x**2, label='quadratic')  # Plot a quadratic line\nplt.plot(x, x**3, label='cubic')  # Plot a cubic line\n\nplt.xlabel('x label')\nplt.ylabel('y label')\nplt.title(\"Multiple Line Plot\")\nplt.legend()  # Add a legend to differentiate between the lines\n\nplt.show()\n```\n\nIn this example, we plot three lines (linear, quadratic, and cubic) on the same plot using `plt.plot()` multiple times. The `label` parameter in each `plt.plot()` call assigns a label to each line, and `plt.legend()` adds a legend to the plot to identify each line.",
        "Question": "12. How can you create a plot with multiple lines in Matplotlib?"
    },
    {
        "Answer": "To sort data in a plot using the `zip()` function, you can follow these steps:\n\n1. First, you need to zip the data you want to sort. For example, if you have two lists `x_values` and `y_values`, you can zip them together using `zip(x_values, y_values)`.\n\n2. Then, you can sort the zipped data based on a specific criterion. You can use the `sorted()` function with a lambda function as the key to specify the sorting criterion. For example, `sorted(zip(x_values, y_values), key=lambda x: x[1], reverse=True)` will sort the zipped data based on the second element of each tuple in descending order.\n\n3. After sorting, you can extract the sorted data back into separate lists. For example, you can extract the sorted `x_values` and `y_values` from the sorted zipped data.\n\nBy following these steps, you can effectively sort data in a plot using the `zip()` function in Python.",
        "Question": "13. How can you sort data in a plot using the zip() function?"
    },
    {
        "Answer": "The lambda function in the sorting code provided in the lecture is used to specify the key for sorting the data. In this case, the lambda function is sorting the data based on the third element of the tuple (x[2]), which corresponds to the math scores. The lambda function allows for custom sorting criteria without defining a separate function.",
        "Question": "14. What is the purpose of the lambda function in the sorting code provided in the lecture?"
    },
    {
        "Answer": "To modify the code to visualize both reading and math scores in a bar chart, you can follow these steps:\n\n1. Gather the values for both reading and math scores for each state.\n2. Plot the reading and math scores together for each state on the bar chart.\n\nHere's a general outline of how you can modify the code:\n\n1. Extract the math scores in addition to the reading scores from the dataset.\n2. Sort the data based on math scores.\n3. Plot both reading and math scores for each state on the bar chart.\n\nYou can use the existing code structure and adapt it to include math scores alongside reading scores in the visualization.",
        "Question": "15. How can you modify the code to visualize both reading and math scores in a bar chart?"
    },
    {
        "Answer": "Advantages of using Matplotlib for data visualization:\n1. **Widely Used**: Matplotlib is one of the most widely used data visualization libraries in Python.\n2. **Flexibility**: It offers a high level of flexibility in creating various types of plots and charts.\n3. **Customization**: Matplotlib allows for extensive customization of plots, including colors, labels, and styles.\n4. **Integration**: It integrates well with other libraries like NumPy and Pandas for data manipulation.\n5. **Open Source**: Being open source, Matplotlib is free to use and has a large community for support and resources.\n6. **Extensive Documentation**: It has extensive documentation and examples available, making it easier to learn and use.\n\nDisadvantages of using Matplotlib for data visualization:\n1. **Steep Learning Curve**: Matplotlib can have a steep learning curve for beginners due to its extensive functionality.\n2. **Complex Syntax**: The syntax for creating complex plots can be verbose and sometimes challenging to remember.\n3. **Aesthetics**: While it is highly customizable, creating aesthetically pleasing plots may require additional effort compared to other libraries.\n4. **Interactive Plots**: Creating interactive plots in Matplotlib is not as straightforward as in some other visualization libraries.\n5. **Performance**: For large datasets, Matplotlib may not be as performant as some other specialized visualization libraries.\n\nThese points can vary based on individual preferences and requirements.",
        "Question": "16. What are the advantages and disadvantages of using Matplotlib for data visualization?"
    }
]
def extract_feature_names(qa_pairs):
    ideal_answers = [pair["Answer"] for pair in qa_pairs]
    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(ideal_answers) 
    feature_names = vectorizer.get_feature_names_out()
    return feature_names.tolist()


# def extract_keywords(text, tfidf_vectorizer, top_n=5):
#     """Identifies keywords using a basic TF-IDF approach."""
#     stop_words = set(stopwords.words('english'))
#     words = word_tokenize(text)  
#     filtered_words = [w for w in words if w.lower() not in stop_words and w.isalpha()]

#     pos_tags = nltk.pos_tag(filtered_words)
#     nouns_verbs = [word for word, pos in pos_tags if pos.startswith('NN') or pos.startswith('VB')]

#     vector = tfidf_vectorizer.transform([" ".join(nouns_verbs)])  
#     sorted_tfidf_scores = vector.toarray()[0].argsort()[-top_n:][::-1] 
#     return [feature_names[index] for index in sorted_tfidf_scores] 

# # Example

# def filter_answers_by_keywords(all_answers, keywords):
#     """Narrow down the list of answers based on keyword presence."""
#     filtered_answers = []
#     for answer in all_answers:
#         if any(word in answer for word in keywords):
#             filtered_answers.append(answer)
#     return filtered_answers

# # Usage before your answer similarity search
# filtered_answers = filter_answers_by_keywords(ideal_answers, student_keywords) 
# print(filtered_answers)