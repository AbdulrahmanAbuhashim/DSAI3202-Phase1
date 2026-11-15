import sqlite3


class DuplicateTracker:
    """Track collision identifiers across processing batches."""

    def __init__(self, database_path):
        self.connection = sqlite3.connect(str(database_path))
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS seen_ids "
            "(collision_index TEXT PRIMARY KEY)"
        )

    def check_batch(self, collision_ids):
        duplicate_flags = []

        for value in collision_ids:
            if value is None or str(value).strip() == "":
                duplicate_flags.append(False)
                continue

            key = str(value).strip()

            try:
                self.connection.execute(
                    "INSERT INTO seen_ids (collision_index) VALUES (?)",
                    (key,)
                )
                duplicate_flags.append(False)
            except sqlite3.IntegrityError:
                duplicate_flags.append(True)

        self.connection.commit()
        return duplicate_flags

    def close(self):
        self.connection.close()
