from ann_utils import *

trainer = ModelTraining('breast_cancer_dataset.xlsx') 

# model ve history(eğitim süresince oluşan train ve val için loss ve değerlendirme metriklerini içerir) oluştur
#ann_model, history = trainer.create_ann_model()

# oluşturulan model ve history kaydet
#trainer.save_model_and_history(ann_model, history, 'saved_models/model1.h5', 'saved_histories/history1.json') 

# kaydedilen model ve history yükle
loaded_ann_model, loaded_history = trainer.load_model_and_history('saved_models/model1.h5', 'saved_histories/history1.json') 

# kaydedilen modelin değerlendirme metriklerini(accuracy, precision gibi) yazdır
trainer.calculate_evaluation_metrics(loaded_ann_model)

# kaydedilen modelin eğitim süresince train ve val için loss ve değerlendirme metriklerinin nasıl ilerlediğini çizdir
metrics = ['loss', 'accuracy', 'precision', 'recall', 'specificity', 'F1_Score', 'auc', 'cohen_kappa']
trainer.plot_evaluation_metrics(loaded_history, metrics)

# kaydedilen modelin karmaşıklık matrisini çizdir
trainer.plot_conf_matrix(loaded_ann_model)
# kaydedilen modelin roc eğrisini çizdir
trainer.plot_roc_curve(loaded_ann_model)