MLFlow Model Manager
=============

.. automodule:: utils.mlflow_model_manager
   :members:
   :undoc-members:
   :show-inheritance:

Classes
-------

.. autoclass:: utils.mlflow_model_manager.MLflowModelManager
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

   .. automethod:: log_model
   
   .. automethod:: load_model
   
   .. automethod:: set_alias
   
   .. automethod:: delete_alias
   
   .. automethod:: list_versions 

   .. automethod:: get_model_by_alias

   .. automethod:: promote_to_production

   .. automethod:: promote_to_staging

   .. automethod:: delete_version

   .. automethod:: update_description
