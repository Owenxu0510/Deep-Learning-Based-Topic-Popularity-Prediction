import fasttext
import codecs
import xlwt


#train model
#classifier = fasttext.supervised('newtrain.txt', 'fasttext300.model', label_prefix="__label__", lr = 0.1, dim=300)

#use trained model
classifier = fasttext.load_model('fasttext50.model.bin', label_prefix="__label__")

result = classifier.test('newtest-order.txt')
print(result.precision)
