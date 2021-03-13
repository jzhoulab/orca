.. Orca documentation master file, created by
   sphinx-quickstart on Wed Mar  3 16:51:28 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Orca's API documentation!
================================

This is the API documentaton of Orca, a deep learning sequence model 
framework for multiscale prediction of genome interactions. 


Most Orca prediction functionalities including structural 
variant effect predictions can be accessed with the **orca_predict** module. 
The **orca_utils** module provide some supporting functions such as structural variant
represention and plotting functionalities. The **orca_model** and **orca_modules** contain the
class definitions for the models and do not need to be directly accessed. 
The **selene_utils2** module provide basic data handling classes for Orca. 

.. toctree::
   :maxdepth: 4
   
   orca_predict
   orca_utils
   orca_models
   orca_modules
   selene_utils2




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
