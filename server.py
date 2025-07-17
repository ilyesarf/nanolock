import grpc
from concurrent import futures
import nanolock_pb2
import nanolock_pb2_grpc
from recognizer import Verification, NoFaceDetected  # put your class in verifier.py

verifier = Verification()

class NanoLockServicer(nanolock_pb2_grpc.NanoLockServicer):
    def Verify(self, request, context):
        user_id = request.user_id
        image_data = request.image_data

        try:
            result = verifier.accept_login(user_id, image_data)
            return nanolock_pb2.FaceResponse(
                verified=result,
                reason="Face matched" if result else "Face mismatch"
            )

        except NoFaceDetected:
            return nanolock_pb2.FaceResponse(
                verified=False,
                reason="No face detected"
            )

        except Exception as e:
            return nanolock_pb2.FaceResponse(
                verified=False,
                reason=f"Error: {str(e)}"
            )

    def AddUser(self, request, context):
        user_id = request.user_id
        image_data = request.image_data
        try:
            verifier.add_face(user_id, image_data)
            # You should also add the user to your DB here if needed
            return nanolock_pb2.AddUserResponse(
                success=True,
                reason="User added successfully"
            )
        except NoFaceDetected:
            return nanolock_pb2.AddUserResponse(
                success=False,
                reason="No face detected"
            )
        except Exception as e:
            return nanolock_pb2.AddUserResponse(
                success=False,
                reason=f"Error: {str(e)}"
            )
        
def serve():
    print("Starting NanoLock Engine...")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    nanolock_pb2_grpc.add_NanoLockServicer_to_server(NanoLockServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("NanoLock Engine running on port 50051...")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
