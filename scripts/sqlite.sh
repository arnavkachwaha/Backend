echo "Running migrations..."
python manage.py makemigrations Cyclops
python manage.py migrate

echo "Opening Django shell..."
python manage.py shell <<EOF
from Cyclops.models import PLRResult
from uuid import UUID
from pprint import pprint

print("=== All PLRResult entries ===")
pprint(list(PLRResult.objects.all()))

print("=== Most recent PLRResult ===")
latest = PLRResult.objects.latest('timestamp')
pprint(vars(latest))

EOF

echo "✅ Done."