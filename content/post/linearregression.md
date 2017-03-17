+++
date = "2017-03-16T15:54:54+05:30"
title = "Linear Regression in easy way"
#categories = [""]
tags = ["hugo", "github"]
summary = "Linear regression"
+++

**Gradient descent algorithm:**  The slope of the convex $\mathcal{Cost Function}$ is subtraced from the parameter with a step size to move towards the minima

$x$ = $(x-\alpha\nabla\mathcal{f(x)})$

Here the move is $\alpha\nabla\mathcal{f(x)}$

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# Load the diabetes dataset
diabetes = datasets.load_diabetes()
```


```python
datasets.load_diabetes?

```


```python
import numpy as np
import random

# m denotes the number of examples here, not the number of features
def gradientDescent(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        # update
        theta = theta - alpha * gradient
    return theta


def genData(numPoints, bias, variance):
    x = np.zeros(shape=(numPoints, 2))
    y = np.zeros(shape=numPoints)
    # basically a straight line
    for i in range(0, numPoints):
        # bias feature
        x[i][0] = 1
        x[i][1] = i
        # our target variable
        y[i] = (i + bias) + random.uniform(0, 1) * variance
    return x, y

# gen 100 points with a bias of 25 and 10 variance as a bit of noise
x, y = genData(100, 25, 10)
m, n = np.shape(x)
numIterations= 100000
alpha = 0.0005
theta = np.ones(n)
theta = gradientDescent(x, y, theta, alpha, m, numIterations)
print(theta)
```



```python


# loading necessary libraries and setting up plotting libraries
import numpy as np
import seaborn as sns

import bokeh.plotting as bp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from sklearn.datasets.samples_generator import make_regression 
from scipy import stats 
from bokeh.models import  WheelZoomTool, ResetTool, PanTool
from JSAnimation import IPython_display

W = 590
H = 350
bp.output_notebook()

%matplotlib inline


```



    <div class="bk-root">
        <a href="http://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
        <span id="56301a16-5da9-4421-8a1a-3dab648f976c">Loading BokehJS ...</span>
    </div>





```python
x = np.linspace(-15,21,1000)
```


```python
np.linspace?

```


```python
x = np.linspace(-15,21,100)
y = x**2-6*x+5

TOOLS = [WheelZoomTool(), ResetTool(), PanTool()]

s1 = bp.figure(width=W, plot_height=H, 
               title='Local minimum of function',  
               )
s1.line(x, y, color="navy", alpha = .5, line_width=3)
s1.circle(3, -4, size =10, color="orange")
s1.title.text_font_size = '16pt'
s1.yaxis.axis_label_text_font_size = "14pt"
s1.xaxis.axis_label_text_font_size = "14pt"

bp.show(s1)
```




    <div class="bk-root">
        <div class="bk-plotdiv" id="2ea08852-0a94-4749-92c6-f9ba045f8342"></div>
    </div>
<script type="text/javascript">
  
  (function(global) {
    function now() {
      return new Date();
    }
  
    var force = false;
  
    if (typeof (window._bokeh_onload_callbacks) === "undefined" || force === true) {
      window._bokeh_onload_callbacks = [];
      window._bokeh_is_loading = undefined;
    }
  
  
    
    if (typeof (window._bokeh_timeout) === "undefined" || force === true) {
      window._bokeh_timeout = Date.now() + 0;
      window._bokeh_failed_load = false;
    }
  
    var NB_LOAD_WARNING = {'data': {'text/html':
       "<div style='background-color: #fdd'>\n"+
       "<p>\n"+
       "BokehJS does not appear to have successfully loaded. If loading BokehJS from CDN, this \n"+
       "may be due to a slow or bad network connection. Possible fixes:\n"+
       "</p>\n"+
       "<ul>\n"+
       "<li>re-rerun `output_notebook()` to attempt to load from CDN again, or</li>\n"+
       "<li>use INLINE resources instead, as so:</li>\n"+
       "</ul>\n"+
       "<code>\n"+
       "from bokeh.resources import INLINE\n"+
       "output_notebook(resources=INLINE)\n"+
       "</code>\n"+
       "</div>"}};
  
    function display_loaded() {
      if (window.Bokeh !== undefined) {
        document.getElementById("2ea08852-0a94-4749-92c6-f9ba045f8342").textContent = "BokehJS successfully loaded.";
      } else if (Date.now() < window._bokeh_timeout) {
        setTimeout(display_loaded, 100)
      }
    }
  
    function run_callbacks() {
      window._bokeh_onload_callbacks.forEach(function(callback) { callback() });
      delete window._bokeh_onload_callbacks
      console.info("Bokeh: all callbacks have finished");
    }
  
    function load_libs(js_urls, callback) {
      window._bokeh_onload_callbacks.push(callback);
      if (window._bokeh_is_loading > 0) {
        console.log("Bokeh: BokehJS is being loaded, scheduling callback at", now());
        return null;
      }
      if (js_urls == null || js_urls.length === 0) {
        run_callbacks();
        return null;
      }
      console.log("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
      window._bokeh_is_loading = js_urls.length;
      for (var i = 0; i < js_urls.length; i++) {
        var url = js_urls[i];
        var s = document.createElement('script');
        s.src = url;
        s.async = false;
        s.onreadystatechange = s.onload = function() {
          window._bokeh_is_loading--;
          if (window._bokeh_is_loading === 0) {
            console.log("Bokeh: all BokehJS libraries loaded");
            run_callbacks()
          }
        };
        s.onerror = function() {
          console.warn("failed to load library " + url);
        };
        console.log("Bokeh: injecting script tag for BokehJS library: ", url);
        document.getElementsByTagName("head")[0].appendChild(s);
      }
    };var element = document.getElementById("2ea08852-0a94-4749-92c6-f9ba045f8342");
    if (element == null) {
      console.log("Bokeh: ERROR: autoload.js configured with elementid '2ea08852-0a94-4749-92c6-f9ba045f8342' but no matching script tag was found. ")
      return false;
    }
  
    var js_urls = [];
  
    var inline_js = [
      function(Bokeh) {
        (function() {
          var fn = function() {
            var docs_json = {"f0530847-96ac-4c86-8383-16eaefef7843":{"roots":{"references":[{"attributes":{},"id":"9b614461-667e-43c4-9b2b-9b8a7b4801e6","type":"BasicTicker"},{"attributes":{},"id":"694b5ec3-074c-451a-9dde-fb1b4961d74a","type":"BasicTickFormatter"},{"attributes":{"bottom_units":"screen","fill_alpha":{"value":0.5},"fill_color":{"value":"lightgrey"},"left_units":"screen","level":"overlay","line_alpha":{"value":1.0},"line_color":{"value":"black"},"line_dash":[4,4],"line_width":{"value":2},"plot":null,"render_mode":"css","right_units":"screen","top_units":"screen"},"id":"93f1a4f7-8905-4edb-9785-fa34ced39a52","type":"BoxAnnotation"},{"attributes":{},"id":"d349d2bd-8e12-40f9-b750-582b9cb04478","type":"ToolEvents"},{"attributes":{"overlay":{"id":"93f1a4f7-8905-4edb-9785-fa34ced39a52","type":"BoxAnnotation"},"plot":{"id":"99f7f7aa-43b4-4a0c-b206-08b1aa9c031c","subtype":"Figure","type":"Plot"}},"id":"777c07ab-b823-4415-a03f-b94552ab462c","type":"BoxZoomTool"},{"attributes":{"axis_label_text_font_size":{"value":"14pt"},"formatter":{"id":"b8bb1121-362f-4450-89c8-aaba0512ca1a","type":"BasicTickFormatter"},"plot":{"id":"99f7f7aa-43b4-4a0c-b206-08b1aa9c031c","subtype":"Figure","type":"Plot"},"ticker":{"id":"9b614461-667e-43c4-9b2b-9b8a7b4801e6","type":"BasicTicker"}},"id":"a5937d8c-c7c7-42c7-bc43-ffd1f66092a9","type":"LinearAxis"},{"attributes":{"below":[{"id":"a5937d8c-c7c7-42c7-bc43-ffd1f66092a9","type":"LinearAxis"}],"left":[{"id":"3d6d0855-ab02-4583-b739-891a1d71b9e0","type":"LinearAxis"}],"plot_height":350,"plot_width":590,"renderers":[{"id":"a5937d8c-c7c7-42c7-bc43-ffd1f66092a9","type":"LinearAxis"},{"id":"9e732a4a-ad61-4510-914c-2c107fdb006f","type":"Grid"},{"id":"3d6d0855-ab02-4583-b739-891a1d71b9e0","type":"LinearAxis"},{"id":"b189f27c-ae32-48b3-99f8-52484b4f8282","type":"Grid"},{"id":"93f1a4f7-8905-4edb-9785-fa34ced39a52","type":"BoxAnnotation"},{"id":"cddd3321-7b20-4476-911f-e7614712755e","type":"GlyphRenderer"},{"id":"fae69d78-f633-4878-9455-2bc1a235cd49","type":"GlyphRenderer"}],"title":{"id":"4945ac7b-6628-4af4-91bf-4045991045af","type":"Title"},"tool_events":{"id":"d349d2bd-8e12-40f9-b750-582b9cb04478","type":"ToolEvents"},"toolbar":{"id":"bb42790a-22a8-49b9-bf22-0595f86c0a82","type":"Toolbar"},"x_range":{"id":"c3f2689d-fab1-4bc0-a278-fd19387cdb95","type":"DataRange1d"},"y_range":{"id":"a152d6fd-12ae-4fed-8843-558f72288cac","type":"DataRange1d"}},"id":"99f7f7aa-43b4-4a0c-b206-08b1aa9c031c","subtype":"Figure","type":"Plot"},{"attributes":{"plot":{"id":"99f7f7aa-43b4-4a0c-b206-08b1aa9c031c","subtype":"Figure","type":"Plot"},"ticker":{"id":"9b614461-667e-43c4-9b2b-9b8a7b4801e6","type":"BasicTicker"}},"id":"9e732a4a-ad61-4510-914c-2c107fdb006f","type":"Grid"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":10},"x":{"value":3},"y":{"value":-4}},"id":"e8d93385-d8e8-4fa6-8464-e912a2c9d490","type":"Circle"},{"attributes":{"data_source":{"id":"cfaba73f-d039-468a-bdac-36ea426c55bb","type":"ColumnDataSource"},"glyph":{"id":"203d5f2f-7f57-4e49-b6d8-076786c44945","type":"Circle"},"hover_glyph":null,"nonselection_glyph":{"id":"e8d93385-d8e8-4fa6-8464-e912a2c9d490","type":"Circle"},"selection_glyph":null},"id":"fae69d78-f633-4878-9455-2bc1a235cd49","type":"GlyphRenderer"},{"attributes":{"plot":{"id":"99f7f7aa-43b4-4a0c-b206-08b1aa9c031c","subtype":"Figure","type":"Plot"}},"id":"3e93d731-0a75-42c3-b16e-ad7667f6562a","type":"HelpTool"},{"attributes":{"line_alpha":{"value":0.5},"line_color":{"value":"navy"},"line_width":{"value":3},"x":{"field":"x"},"y":{"field":"y"}},"id":"942db697-018e-4e1d-93bc-bb36f33f72a0","type":"Line"},{"attributes":{"fill_color":{"value":"orange"},"line_color":{"value":"orange"},"size":{"units":"screen","value":10},"x":{"value":3},"y":{"value":-4}},"id":"203d5f2f-7f57-4e49-b6d8-076786c44945","type":"Circle"},{"attributes":{"plot":{"id":"99f7f7aa-43b4-4a0c-b206-08b1aa9c031c","subtype":"Figure","type":"Plot"}},"id":"4829d460-4f33-4e79-9cdd-6fae63af3b5b","type":"ResetTool"},{"attributes":{"axis_label_text_font_size":{"value":"14pt"},"formatter":{"id":"694b5ec3-074c-451a-9dde-fb1b4961d74a","type":"BasicTickFormatter"},"plot":{"id":"99f7f7aa-43b4-4a0c-b206-08b1aa9c031c","subtype":"Figure","type":"Plot"},"ticker":{"id":"cce613fa-d532-41cf-9243-ae1c4fd1ebb9","type":"BasicTicker"}},"id":"3d6d0855-ab02-4583-b739-891a1d71b9e0","type":"LinearAxis"},{"attributes":{"plot":null,"text":"Local minimum of function","text_font_size":{"value":"16pt"}},"id":"4945ac7b-6628-4af4-91bf-4045991045af","type":"Title"},{"attributes":{},"id":"b8bb1121-362f-4450-89c8-aaba0512ca1a","type":"BasicTickFormatter"},{"attributes":{"plot":{"id":"99f7f7aa-43b4-4a0c-b206-08b1aa9c031c","subtype":"Figure","type":"Plot"}},"id":"d548d195-3539-4217-be93-fd02c27d65dd","type":"PanTool"},{"attributes":{},"id":"cce613fa-d532-41cf-9243-ae1c4fd1ebb9","type":"BasicTicker"},{"attributes":{"active_drag":"auto","active_scroll":"auto","active_tap":"auto","tools":[{"id":"d548d195-3539-4217-be93-fd02c27d65dd","type":"PanTool"},{"id":"ecce127b-65bb-43cf-bad9-41b06ece9ae7","type":"WheelZoomTool"},{"id":"777c07ab-b823-4415-a03f-b94552ab462c","type":"BoxZoomTool"},{"id":"d5f61b91-d24d-4603-997d-aaaa406e0457","type":"SaveTool"},{"id":"4829d460-4f33-4e79-9cdd-6fae63af3b5b","type":"ResetTool"},{"id":"3e93d731-0a75-42c3-b16e-ad7667f6562a","type":"HelpTool"}]},"id":"bb42790a-22a8-49b9-bf22-0595f86c0a82","type":"Toolbar"},{"attributes":{"callback":null},"id":"a152d6fd-12ae-4fed-8843-558f72288cac","type":"DataRange1d"},{"attributes":{"dimension":1,"plot":{"id":"99f7f7aa-43b4-4a0c-b206-08b1aa9c031c","subtype":"Figure","type":"Plot"},"ticker":{"id":"cce613fa-d532-41cf-9243-ae1c4fd1ebb9","type":"BasicTicker"}},"id":"b189f27c-ae32-48b3-99f8-52484b4f8282","type":"Grid"},{"attributes":{"callback":null},"id":"cfaba73f-d039-468a-bdac-36ea426c55bb","type":"ColumnDataSource"},{"attributes":{"callback":null,"column_names":["x","y"],"data":{"x":{"__ndarray__":"AAAAAAAALsBGF1100UUtwIwuuuiiiyzA0kUXXXTRK8AXXXTRRRcrwF100UUXXSrAo4suuuiiKcDpoosuuugowC666KKLLijAdNFFF110J8C66KKLLromwAAAAAAAACbARhdddNFFJcCMLrroooskwNJFF1100SPAF1100UUXI8BddNFFF10iwKOLLrrooiHA6KKLLrroIMAuuuiiiy4gwOiiiy666B7AdNFFF110HcAAAAAAAAAcwIwuuuiiixrAGF100UUXGcCiiy666KIXwC666KKLLhbAuuiiiy66FMBGF1100UUTwNJFF1100RHAXHTRRRddEMDQRRdddNENwOiiiy666ArAAAAAAAAACMAYXXTRRRcFwCy66KKLLgLAiC666KKL/r+46KKLLrr4v+iiiy666PK/MLrooosu6r8AXXTRRRfdvwAXXXTRRbe/gNFFF1100T9gdNFFF13kPwAAAAAAAPA/0EUXXXTR9T+giy666KL7P7jooosuugBAoIsuuuiiA0CQLrrooosGQHjRRRdddAlAYHTRRRddDEBIF1100UUPQBhddNFFFxFAjC666KKLEkAAAAAAAAAUQHTRRRdddBVA6KKLLrroFkBcdNFFF10YQNRFF1100RlASBdddNFFG0C86KKLLrocQDC66KKLLh5ApIsuuuiiH0CMLrrooosgQEYXXXTRRSFAAAAAAAAAIkC66KKLLroiQHTRRRdddCNALrrooosuJEDqoosuuugkQKSLLrrooiVAXnTRRRddJkAYXXTRRRcnQNJFF1100SdAjC666KKLKEBGF1100UUpQAAAAAAAACpAuuiiiy66KkB00UUXXXQrQDC66KKLLixA6qKLLrroLECkiy666KItQF500UUXXS5AGF100UUXL0DSRRdddNEvQEYXXXTRRTBAo4suuuiiMEAAAAAAAAAxQF500UUXXTFAuuiiiy66MUAYXXTRRRcyQHTRRRdddDJA0kUXXXTRMkAuuuiiiy4zQIwuuuiiizNA6KKLLrroM0BGF1100UU0QKSLLrroojRAAAAAAAAANUA=","dtype":"float64","shape":[100]},"y":{"__ndarray__":"AAAAAAAAdEACN5ZBqTBzQKxnh8CNZXJA/JHTfK2ecUDztXp2CNxwQJLTfK2eHXBAsNWzQ+DGbkCK9yOn+VptQLAMSoWJ92tAKBUm3o+cakDtELixDEppQAAAAAAAAGhAYuL9yGm+ZkASuLEMSoVlQBCBG8ugVGRAWj07BG4sY0D27BC4sQxiQN6PnOZr9WBAKEy8HznNX0A0X6tnh8BdQNtYBqXCxFtAIDnN1+rZWUAAAAAAAABYQH6tnh0CN1ZAl0GpMPF+VEBMvB85zddSQJ4dAjeWQVFAG8ugVJh4T0AzKBUm3o9MQIRSYeL9yElAC0qFifcjR0DODoEby6BEQMqgVJh4P0JAAAAAAAAAQEDdWAalwsQ7QCdMvB85zTdA6NkhcGMZNEAcAjeWQakwQISJ9yOn+SpAtEPgxjIoJUCEZVAqTLwfQHatnh0CNxZAYr5Wzw6BC0DQ6tkhcGP5PwAAAAAAAAAAsEPgxjIo9b8ecGMZlAoDwNTq2SFwYwnA+JHTfK2eDcCQZVAqTLwPwIxlUCpMvA/A+JHTfK2eDcDU6tkhcGMJwCBwYxmUCgPAsEPgxjIo9b8AAAAAAAAAANDq2SFwY/k/YL5Wzw6BC0B4rZ4dAjcWQKBlUCpMvB9AvEPgxjIoJUCMifcjp/kqQCACN5ZBqTBA7tkhcGMZNEAsTLwfOc03QN5YBqXCxDtAAAAAAAAAQEDLoFSYeD9CQM4OgRvLoERAC0qFifcjR0CHUmHi/chJQDgoFSbej0xAIMugVJh4T0CgHQI3lkFRQEy8HznN11JAl0GpMPF+VEB+rZ4dAjdWQAAAAAAAAFhAHjnN1+rZWUDbWAalwsRbQDhfq2eHwF1ALky8HznNX0Dgj5zma/VgQPfsELixDGJAXD07BG4sY0APgRvLoFRkQBK4sQxKhWVAYuL9yGm+ZkAAAAAAAABoQPAQuLEMSmlAKBUm3o+cakC0DEqFifdrQIn3I6f5Wm1AstWzQ+DGbkCR03ytnh1wQPS1enYI3HBA+pHTfK2ecUCsZ4fAjWVyQAQ3lkGpMHNAAAAAAAAAdEA=","dtype":"float64","shape":[100]}}},"id":"8eef6c0f-a40c-4140-879e-ae4f687c7b9a","type":"ColumnDataSource"},{"attributes":{"callback":null},"id":"c3f2689d-fab1-4bc0-a278-fd19387cdb95","type":"DataRange1d"},{"attributes":{"plot":{"id":"99f7f7aa-43b4-4a0c-b206-08b1aa9c031c","subtype":"Figure","type":"Plot"}},"id":"d5f61b91-d24d-4603-997d-aaaa406e0457","type":"SaveTool"},{"attributes":{"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"line_width":{"value":3},"x":{"field":"x"},"y":{"field":"y"}},"id":"4ee26159-eaee-433c-83a5-fdf9953699cf","type":"Line"},{"attributes":{"plot":{"id":"99f7f7aa-43b4-4a0c-b206-08b1aa9c031c","subtype":"Figure","type":"Plot"}},"id":"ecce127b-65bb-43cf-bad9-41b06ece9ae7","type":"WheelZoomTool"},{"attributes":{"data_source":{"id":"8eef6c0f-a40c-4140-879e-ae4f687c7b9a","type":"ColumnDataSource"},"glyph":{"id":"942db697-018e-4e1d-93bc-bb36f33f72a0","type":"Line"},"hover_glyph":null,"nonselection_glyph":{"id":"4ee26159-eaee-433c-83a5-fdf9953699cf","type":"Line"},"selection_glyph":null},"id":"cddd3321-7b20-4476-911f-e7614712755e","type":"GlyphRenderer"}],"root_ids":["99f7f7aa-43b4-4a0c-b206-08b1aa9c031c"]},"title":"Bokeh Application","version":"0.12.4"}};
            var render_items = [{"docid":"f0530847-96ac-4c86-8383-16eaefef7843","elementid":"2ea08852-0a94-4749-92c6-f9ba045f8342","modelid":"99f7f7aa-43b4-4a0c-b206-08b1aa9c031c"}];
            
            Bokeh.embed.embed_items(docs_json, render_items);
          };
          if (document.readyState != "loading") fn();
          else document.addEventListener("DOMContentLoaded", fn);
        })();
      },
      function(Bokeh) {
      }
    ];
  
    function run_inline_js() {
      
      if ((window.Bokeh !== undefined) || (force === true)) {
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i](window.Bokeh);
        }if (force === true) {
          display_loaded();
        }} else if (Date.now() < window._bokeh_timeout) {
        setTimeout(run_inline_js, 100);
      } else if (!window._bokeh_failed_load) {
        console.log("Bokeh: BokehJS failed to load within specified timeout.");
        window._bokeh_failed_load = true;
      } else if (force !== true) {
        var cell = $(document.getElementById("2ea08852-0a94-4749-92c6-f9ba045f8342")).parents('.cell').data().cell;
        cell.output_area.append_execute_result(NB_LOAD_WARNING)
      }
  
    }
  
    if (window._bokeh_is_loading === 0) {
      console.log("Bokeh: BokehJS loaded, going straight to plotting");
      run_inline_js();
    } else {
      load_libs(js_urls, function() {
        console.log("Bokeh: BokehJS plotting callback run at", now());
        run_inline_js();
      });
    }
  }(this));
</script>



```python
old_min = 0
temp_min = 15
step_size = 0.01
precision = 0.001
 
def f_derivative(x):
    return 2*x -6

mins = []
cost = []

while abs(temp_min - old_min) > precision:
    old_min = temp_min 
    gradient = f_derivative(old_min) 
    move = gradient * step_size
    temp_min = old_min - move
 
print("Local minimum occurs at {}.".format(round(temp_min,2)))


```

    Local minimum occurs at 3.05.





```python

```
