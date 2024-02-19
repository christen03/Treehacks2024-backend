from convex import ConvexClient
from dotenv import load_dotenv
import os

load_dotenv()

convex_client = os.getenv("CONVEX_CLIENT")
print("convex_client", convex_client)
convex = ConvexClient(convex_client)

def get_student_from_phone_number(student_phone_number):
    return convex.query("students:getStudentFromPhoneNumber", {'phoneNumber': student_phone_number})

def update_student_has_seen_array(userid, index):
    return convex.mutation("students:updateStudentsHasSeenArray", 
                           {'studentId': userid, 'index': index})

def increment_texts_responded_and_score(userid, score, texts):
    return convex.mutation("students:updateStudentsTextsResponded", 
                           {'studentId': userid, 'currentScore': score})

def get_number_of_texts_responded(userid):
    return convex.query("students:getNumberOfTextsStudentHasSeen", {'studentId': userid})