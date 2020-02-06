from rest_framework.response import Response
from rest_framework.views import status


def validate_request_data(fn):
    def decorated(*args, **kwargs):
        # args[0] == GenericView Object
        productId = args[0].request.data.get("productId", "")
        userId = args[0].request.data.get("userId", "")
        if not productId or not userId:
            return Response(
                data={
                    "message": "[Unsuccessful] Both productId and userId are required to add a product"
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        return fn(*args, **kwargs)
    return decorated