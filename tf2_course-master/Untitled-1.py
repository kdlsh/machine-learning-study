# To add a new cell, type '#%%'
#%%
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'machine-learning-study\\tf2_course-master'))
	print(os.getcwd())
except:
	pass

