# Install required packages
# pip install plaid-python

# Create financial_connectors.py
from plaid.model.country_code import CountryCode
from plaid.model.products import Products
from plaid.api import plaid_api
from plaid.model.link_token_create_request import LinkTokenCreateRequest
from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest
import plaid

class PlaidConnector:
    def __init__(self, client_id, secret, environment):
        configuration = plaid.Configuration(
            host=plaid.Environment.Sandbox,
            api_key={
                'clientId': client_id,
                'secret': secret,
            }
        )
        api_client = plaid.ApiClient(configuration)
        self.client = plaid_api.PlaidApi(api_client)
        
    def create_link_token(self, user_id):
        # Create a link token for the given user
        request = LinkTokenCreateRequest(
            user={"client_user_id": user_id},
            client_name="Financial Health Assistant",
            products=[Products("transactions")],
            country_codes=[CountryCode("US")],
            language="en"
        )
        response = self.client.link_token_create(request)
        return response['link_token']