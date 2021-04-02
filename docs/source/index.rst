.. Orca documentation master file, created by
   sphinx-quickstart on Wed Mar  3 16:51:28 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Orca's API documentation!
================================

This is the API documentaton of Orca, a deep learning sequence model 
framework for multiscale prediction of genome 3D interactions. You can find the Github repository
`here <https://github.com/jzhoulab/orca>`_ and the webserver `here <https://orca.zhoulab.io/>`_.

The **orca_predict** module is the main module that provides various Orca prediction functions including 
functions for predicting structural variant effect for both simple and complex variants. It also provides 
functions for making predictions from genomic regions or directly from any genomic sequences. The **orca_utils** module provides some supporting functions including a class for structural variant
represention and plotting functions. The **orca_model** and **orca_modules** contain the class definitions for the models 
and generally do not need to be directly accessed. The **selene_utils2** module provides `selene <https://github.com/FunctionLab/selene>`_ extensions
that provide basic data handling classes for Orca. 

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
