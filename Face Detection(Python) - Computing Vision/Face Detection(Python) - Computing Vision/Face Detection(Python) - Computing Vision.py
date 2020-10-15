#Εισαγωγή του OpenCV
import cv2 as cv

#Μεταβλητή για την καταγραφή βίντεο από την κάμερα του Υπολογιστή
cap = cv.VideoCapture(0)
cap.set(3, 640) #Αρχικοποίηση Μήκους του Frame
cap.set(4, 480) #Αρχικοιποίηση Ύψους του Frame

#Εισαγωγή των απαραίτητων XML Classifiers αρχείων από το OpenCV
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

while(True):
    #Διάβασμα Frame ανα Frame της εικόνας μέσα από το βίντεο της κάμερας
    ret, frame = cap.read()

    #Αρχικοποιούμε όλο το frame που παίρνουμε από το βίντεο της κάμερας ως Γκρι για την καλύτερη ανίχνευση του προσώπου από τους Haar Classifiers
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #Χρήση του ταξινομητή face_cascace για την ανίχνευση του προσώπου
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #Εμφάνιση της τιμής 1 στην κονσόλα κάθε φορά που ανιχνεύει κάποιο πρόσωπο
    print(len(faces))

    #Δημιουργία και εμφάνιση Πλαισίων για την ανίχνευση του προσώπου και των ματιών
    for (x,y,w,h) in faces:
         #Σχεδιασμός του τετραγώνου ανίχνευσης του προσώπου
         cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),4)  
         #Αρχικοποίηση των περιοχών ενδιαφέροντος(ROI) για τα πλαίσια των ματιών
         roi_gray = gray[y:y+h, x:x+w]  
         roi_color = frame[y:y+h, x:x+w]
         #Χρήση του ταξινομητή eyes_cascace για την εύρεση των ματιών
         eyes = eye_cascade.detectMultiScale(roi_gray)   
         for (ex,ey,ew,eh) in eyes:
              #Σχεδιασμός του τετραγώνου ανίχνευσης των ματιών
             cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),4)
    

    #frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) Εμφάνιση ενός γκρίζου τελικού παραθύρου
    
    #Εμφάνιση του τελικού παραθύρου
    cv.imshow('Face Detection',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

#Όταν κλείνουμε την εφαρμογή, κλέινουμε όλα τα παράθυρα και σταματάμε την καταγραφή από την κάμερα
cap.release()
cv.destroyAllWindows()