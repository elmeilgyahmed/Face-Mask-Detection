
import numpy as np
import math
import pickle
from sklearn.feature_selection import SelectPercentile, f_classif




# Viola Jones Helpers Functions 
'''
    These helpers function to help in the training process , on weak classifers 
1-Integral Image 
2-extract_harr_features
3-apply features 
4- weak_classifer_training  *
5- select_best_weak_classifer *
6- classify *
 .
 .
 .
 
 * asteriks  functions is related to the weak classifers that are classify on only small portion of the image 
 
'''

class ViolaJones:
    def __init__(self, Number_of_weak_classifer = 10):
        
        self.Number_of_weak_classifer = Number_of_weak_classifer
        self.alphas = []
        self.classifers = []

    def train(self, training_data_with_label, positive_num, negitive_num):
   
        weights = np.zeros(len(training_data_with_label))
        training_data = []
        print("Computing integral images")
        for x in range(len(training_data_with_label)):
            training_data.append((integral_image(training_data_with_label[x][0]), training_data_with_label[x][1]))
            if training_data_with_label[x][1] == 1:
                weights[x] = 1.0 / (2 * positive_num)
            else:
                weights[x] = 1.0 / (2 * negitive_num)

        #####Building features
        features = self.build_features(training_data[0][0].shape)
        ## APPLYING these features on each training example
        X, y = self.apply_features(features, training_data)
        ##Selecting the bes classifiers
        indices = SelectPercentile(f_classif, percentile=10).fit(X.T, y).get_support(indices=True)
        X = X[indices]
        features = features[indices]
        ##now selecting the strong features

        for t in range(self.Number_of_weak_classifer):
            weights = weights / np.linalg.norm(weights)
            weak_classifiers = self.train_weak(X, y, features, weights)
            classifier, error, accuracy = self.select_best(weak_classifiers, weights, training_data)
            beta = error / (1.0 - error)
            for i in range(len(accuracy)):
                weights[i] = weights[i] * (beta ** (1 - accuracy[i]))
            alpha = math.log(1.0/beta)
            self.alphas.append(alpha)
            self.classifers.append(classifier)
          

    def train_weak(self, each_train_feature, classification_of_each_Train_example, features, weights):
        
        '''         
         training weak classifer is as following :
        for each classifer is given weight alpha , and output classifet is sign (alpha_1 * clf_1 + alpha_2*clf_2 +alpha_3*clf_3 ... and so on until the specified number of classifers)
        if the sign + face 
        if the sign - not face 
    
        - alphas is the weights of each classifer given by the Adaboost , and it is updatable paramerts based the previous clf 
        - the incorrectly classifed example will be given higher weight so to make the next classifer will train  good on it and so on until the end of classifers 
    
    
        DIAGRAM
    
        clf_1 ---> clf_2 ---> clf_3 --->
          |       /  |       /
          |      /   |      /
          ex i   /   ex j   /
          is    /    is    /
          incorr     incorr
     
        the updatable equtaion of the weight is  
     
        Wi=Wi*beta^(1-ei)            give the incrrectly classified larger weight 
        beta = epsilon/1-epsilon     where the epsilon value  is the error from the target value 
     
        we sholud first now how the classification is done to know how to calculate the error epsilon 
     
        the classification is done as following 
        - for each feature (harr_extracted_feature ) we will define two updatble (learnable ) parameter 
        1. theta ---> which is the threshold 
        2. polarity ---> which is either +1 or -1 
     
        and classification is done from the direct equation 
     
        if p(i)*feature(i) < p(i) * theta(i) 
          predicted_label(i)=1
          else 
          predicated_label(i)=0 
     
        and epsilon the error of the best classifer is calculated as following 
        epsilon=minmun_p_theta_feature(sum_over_all_classifers predicated_label(i)-target(i))
     
    
        '''
        
       
        
    
   
        count_of_positive = 0
        count_of_negitive = 0
        for weight, label in zip(weights, y):
            if label == 1:
                count_of_positive += weight
            else:
                count_of_negitive += weight

        classifiers = []
        total_features = each_train_feature.shape[0]
        for index, feature in enumerate(each_train_feature):
            
            applied_feature = sorted(zip(weights, feature, classification_of_each_Train_example), key=lambda x: x[1])

            positive_seen = 0 
            negitive_seen = 0
            positive_weights= 0 
            negitive_weights = 0
            
            min_error, best_feature, best_threshold, best_polarity = float('inf'), None, None, None
            for w, f, label in applied_feature:
                error = min(negitive_weights + count_of_positive - positive_weights, positive_weights + count_of_negitive - negitive_weights)
                if error < min_error:
                    min_error = error
                    best_feature = features[index]
                    best_threshold = f
                    best_polarity = 1 if positive_seen > negitive_seen else -1

                if label == 1:
                    positive_seen += 1
                    positive_weights += w
                else:
                    negitive_seen += 1
                    negitive_weights += w
            
            classifier = Weak_Classifers(best_feature[0], best_feature[1], best_threshold, best_polarity)
            classifiers.append(classifier)
        return classifiers
                
    def build_features(self, image_shape):
       ## our shifting on matrix (stride) = 1
       
        image_height, image_width = image_shape
        features = []   # [Tuple(Pos-Region,Neg-Region)........)]
        for width in range(1, image_width+1):
            for height in range(1, image_height+1):
                col = 0
                while col + width < image_width:
                    
                    row = 0
                    while row + height < image_height:
                        
                        
                        '''
                        rects_positions
                    
                        immediate-----> # |R|      -----> rect_region(col,row,width,height) 
                    
                        right---------> # | |R|    -----> rect_region(col+width,row,width,height)
                    
                        right_away----> # | | |R|  -----> rect_region(col+2*width,row,width,height)
                    
                        bottom--------> # | |      -----> rect_region(col,row+height,width,height)
                                      |R|
                    
                        bottom_right--> # | |      -----> rect_region(col+width,row+height,width,height)
                                      | | |R|
                    
                        bottom_away--------> # | |      -----> rect_region(col,row+2*height,width,height)
                                           | |
                                           |R|
                    
                        
                    
                    
                        Harr-Features 
                    
                        1- 2-rect
                        2- 3-rect
                        3- 4-rect
                    
                        2-rect
                        W---> White Area
                        D---> Dark Area
                    
                        - W | D    [right---->D,immediate---->W]  Horizental           
                                                   
                    
                    
                        - D        [immediate--->D,bottom----W] Vertical
                        _
                          
                        W
                      
                        3-rect
                        - W | D | W    [[right]---->D,[right_away,immediate]--->W]     Horizental              
                                                   
                    
                    
                        - W        [[bottom]---->D,[bottom_away,immediate]--->W]     Vertical
                        _
                          
                        D
                        _
                      
                        W
                      
                        4-rect
                        W | D
                    
                        _ | _
                        
                        D | W
                    
                        [[right,bottom]---->D,[immediate,bottom_right]--->W]     
                    
                        '''
                        
                   
                    
                        # rect positions
                        immediate = RectangleRegion(col, row, width, height)
                        right = RectangleRegion(col+width, row, width, height)
                        bottom=RectangleRegion(col,row+width,width,height)
                        right_away=RectangleRegion(col+2*width,row,width,height)
                        bottom_away = RectangleRegion(col, row+2*height, width, height)
                        bottom_rightt = RectangleRegion(col+width, row+height, width, height)
                        
                        # BUILDING Harr feature
                        ## 2 rect
                        if col + 2 * width < image_width: 
                            features.append(([right], [immediate]))

                        
                        if row + 2 * height < image_height:
                            features.append(([immediate], [bottom]))
                        
                        
                        #3 rect
                        if col + 3 * width < image_width: 
                            features.append(([right], [right_away, immediate]))

                        
                        if row + 3 * height < image_height: 
                            features.append(([bottom], [bottom_away, immediate]))

                        #4 rect
                       
                        if col + 2 * width < image_width and row + 2 * height < image_height:
                            features.append(([right, bottom], [immediate, bottom_rightt]))

                        row += 1 ##1 is the shift value
                    col += 1      ##1 is the shift value
        return np.array(features) ##return numpy array of features

    def select_best(self, weak_classifiers, weights, integral_images_their_classification):
        """
        
          INPUTS:
            weak_classifiers: array of weak classifiers
            weights: array of weights
            integral_images_their_classification: An array of tuples. The first element is the numpy array of integral images and the second element is its label 1 or 0
          OUTPUT:
            the best classifier, its error, and an array of its best accuracy
        """
        best_classifier = None 
        best_error = float('inf') 
        best_accuracy = None
        for classifier in weak_classifiers:
            error  = 0
            accuracy = [] ## initlaize empty array to append on it the accuracy
            for integral_image_its_label, weights in zip(integral_images_their_classification, weights):
                best_value = abs(classifier.classify(integral_image_its_label[0]) - integral_image_its_label[1])
                accuracy.append(best_value)
                error += weights * best_value
            error = error / len(integral_images_their_classification)
            if error < best_error:
                best_classifier, best_error, best_accuracy = classifier, error, accuracy
        return best_classifier, best_error, best_accuracy
    
    def apply_features(self, Regions_of_pos_neg_contributation, integral_images_their_classification):
 
        ##INItlize the numpy array that will be returned
        applied_features = np.zeros((len(Regions_of_pos_neg_contributation), len(integral_images_their_classification)))
        label = np.array(list(map(lambda data: data[1], integral_images_their_classification)))
        i = 0
        for positive_regions_contribuation, negative_regions_contribuation in Regions_of_pos_neg_contributation:
            feature = lambda ii: sum([pos.compute_feature(ii) for pos in positive_regions_contribuation]) - sum([neg.compute_feature(ii) for neg in negative_regions_contribuation])
            applied_features[i] = list(map(lambda data: Regions_of_pos_neg_contributation(data[0]), integral_images_their_classification))
            i += 1
        return applied_features, label
    ##FUNCTION THAT Responsible for classifying if face or not
    def classify(self, image):
        ##FUNCTION THAT Responsible for classifying if face or not
        '''
        INPUTS:
            THe image
        OUTPUTS:
            1 or 0
            
        '''
        total = 0
        integral_imageeee = integral_image(image)
        for alpha, clf in zip(self.alphas, self.clfs):
            total += alpha * clf.classify(integral_imageeee)
        return 1 if total >= 0.5 * sum(self.alphas) else 0
    
    '''
    ## Load and save the pkl files of weights
    def save(self, filename):
        with open(filename+".pkl", 'wb') as f:
            pickle.dump(self, f)

    def load(filename):
        with open(filename+".pkl", 'rb') as f:
            return pickle.load(f)
    '''
class WeakClassifier:
    def __init__(self, positive_regions, negative_regions, threshold, polarity):
        """
          Instructor variblesssss:
            positive_regions: An array of RectangleRegions which positively contribute to a feature
            negative_regions: An array of RectangleRegions which negatively contribute to a feature
            threshold: The threshold for the weak classifier
            polarity: The polarity of the weak classifier
        """
        self.positive_regions = positive_regions
        self.negative_regions = negative_regions
        self.threshold = threshold
        self.polarity = polarity
    
    def classify(self, x):
    
        feature = lambda ii: sum([pos.compute_feature(ii) for pos in self.positive_regions]) - sum([neg.compute_feature(ii) for neg in self.negative_regions])
        return 1 if self.polarity * feature(x) < self.polarity * self.threshold else 0
    
        
def integral_image(image):
    
    
    '''
        this function to calculate integral image of given image
    
        it is simple function directly from the following equation s
    
    
        integral_image(-1, y) = 0
        cumulative_row_sum(x, -1) = 0
        cumulative_row_sum(x, y) = cumulative_row_sum(x, y-1) + image(x, y)  # Sum of column X at level Y
        integral_image(x, y) = integral_image(x-1, y) + cumulative_row_sum(x, y) 
    '''
    h,w =image.shape
    integral_image = np.zeros(image.shape)
    cumulative_row_sum = np.zeros(image.shape)
    for y in range(h):
        for x in range(w):
            cumulative_row_sum[y][x] = cumulative_row_sum[y-1][x] + image[y][x] if y-1 >= 0 else image[y][x]
            integral_image[y][x] = integral_image[y][x-1]+cumulative_row_sum[y][x] if x-1 >= 0 else cumulative_row_sum[y][x]
    return integral_image



'''
class RectangleRegion:
    def __init__(self, x_of_upper_left_rect, y_of_upper_left_rect, width_of_rectangle, height_of_rectangle):
        self.x_of_upper_left_rect = x_of_upper_left_rect
        self.y_of_upper_left_rect = y_of_upper_left_rect
        self.width_of_rectangle = width_of_rectangle
        self.height_of_rectangle = height_of_rectangle
    
    def compute_feature(self, integ_image):
  
        return integ_image[self.y_of_upper_left_rect+self.height_of_rectangle][self.x_of_upper_left_rect+self.width_of_rectangle] + integ_image[self.y_of_upper_left_rect][self.x_of_upper_left_rect] - (integ_image[self.y_of_upper_left_rect+self.height_of_rectangle][self.x_of_upper_left_rect]+integ_image[self.y_of_upper_left_rect][self.x_of_upper_left_rect+self.width_of_rectangle])
'''
class RectangleRegion:
    def __init__(self, x, y, width, height):
        
        """
        Instructor variabless
        
            x: x coordinate of the upper left of the rectangle
            y: y coordinate of the upper left of the rectangle
            width:  rectangle width
            height: rectangle height
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def compute_feature(self, iimage):
      
        return iimage[self.y+self.height][self.x+self.width] + iimage[self.y][self.x] - (iimage[self.y+self.height][self.x]+iimage[self.y][self.x+self.width])

'''
def train_viola(t):
    with open("training.pkl", 'rb') as f:
        training = pickle.load(f)
    clf = ViolaJones(T=t)
    clf.train(training, 2429, 4548)
    evaluate(clf, training)
    clf.save(str(t))

train_viola(50)

'''

