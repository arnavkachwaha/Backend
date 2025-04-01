import logging
from django.db import transaction, DatabaseError
from myapp.models import MyModel  # Replace with your actual model

logger = logging.getLogger(__name__)

class SQLiteORMService:
    """Service class for handling database operations using Django ORM."""

    @staticmethod
    def get_all():
        """Fetches all records from MyModel."""
        try:
            return MyModel.objects.all()
        except DatabaseError as e:
            logger.error(f"Error fetching records: {e}")
            raise

    @staticmethod
    def get_by_id(record_id):
        """Fetches a single record by ID."""
        try:
            return MyModel.objects.get(id=record_id)
        except MyModel.DoesNotExist:
            logger.warning(f"Record with ID {record_id} not found.")
            return None
        except DatabaseError as e:
            logger.error(f"Error fetching record by ID: {e}")
            raise

    @staticmethod
    def create_record(data):
        """Creates a new record."""
        try:
            record = MyModel.objects.create(**data)
            return record
        except DatabaseError as e:
            logger.error(f"Error creating record: {e}")
            raise

    @staticmethod
    def update_record(record_id, data):
        """Updates an existing record."""
        try:
            record = MyModel.objects.filter(id=record_id).update(**data)
            return record
        except DatabaseError as e:
            logger.error(f"Error updating record: {e}")
            raise

    @staticmethod
    def delete_record(record_id):
        """Deletes a record."""
        try:
            MyModel.objects.filter(id=record_id).delete()
            return True
        except DatabaseError as e:
            logger.error(f"Error deleting record: {e}")
            raise

    @staticmethod
    def bulk_create(records):
        """Bulk inserts multiple records."""
        try:
            MyModel.objects.bulk_create([MyModel(**data) for data in records])
            return True
        except DatabaseError as e:
            logger.error(f"Error in bulk creation: {e}")
            raise

    @staticmethod
    def execute_transaction(operations):
        """Executes multiple operations as an atomic transaction."""
        try:
            with transaction.atomic():
                for operation in operations:
                    operation()
            return True
        except DatabaseError as e:
            logger.error(f"Transaction failed: {e}")
            raise
