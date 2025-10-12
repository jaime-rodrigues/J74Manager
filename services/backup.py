import os
import asyncio
from typing import List
from urllib.parse import urlparse

from core.config import settings
from services.database import DatabaseManager

class BackupManager:
    """Manages asynchronous database backup and restore operations."""
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.db_params = self._get_db_params_from_url(settings.DATABASE_URL)
        os.makedirs(settings.BACKUP_DIR, exist_ok=True)

    def _get_db_params_from_url(self, db_url: str) -> dict:
        result = urlparse(db_url)
        return {
            'dbname': result.path[1:],
            'user': result.username,
            'password': result.password,
            'host': result.hostname,
            'port': result.port
        }

    async def _run_cli_command(self, command: List[str]):
        """Runs a command line process asynchronously."""
        cli_env = os.environ.copy()
        cli_env['PGPASSWORD'] = self.db_params['password']
        
        process = await asyncio.create_subprocess_exec(
            *command,
            env=cli_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"Command failed with exit code {process.returncode}: {stderr.decode()}")
        return stdout.decode()

    async def backup_database(self, filename: str) -> str:
        """Asynchronously dumps the database contents to a file."""
        filepath = os.path.join(settings.BACKUP_DIR, filename)
        command = [
            'pg_dump',
            '--dbname', self.db_params['dbname'],
            '--username', self.db_params['user'],
            '--host', self.db_params['host'],
            '--port', str(self.db_params['port']),
            '--file', filepath,
            '--format', 'c',
            '--no-password'
        ]
        print(f"Starting database backup to {filepath}...")
        await self._run_cli_command(command)
        print("Backup complete.")
        return filepath

    async def restore_database(self, filename: str):
        """Asynchronously restores the database from a backup file."""
        filepath = os.path.join(settings.BACKUP_DIR, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Backup file not found: {filepath}")

        # Close the connection pool before restoring
        await self.db_manager.close_pool()

        command = [
            'pg_restore',
            '--dbname', self.db_params['dbname'],
            '--username', self.db_params['user'],
            '--host', self.db_params['host'],
            '--port', str(self.db_params['port']),
            '--clean',
            '--if-exists',
            '--no-password',
            filepath
        ]
        print(f"Starting database restore from {filepath}...")
        await self._run_cli_command(command)
        
        # Re-create the pool. The connect_pool method now handles the extension check internally.
        await self.db_manager.connect_pool()
        print("Restore complete.")

    def get_latest_backup(self) -> str | None:
        """Finds the most recent backup file (this remains a synchronous operation)."""
        try:
            backup_files = [f for f in os.listdir(settings.BACKUP_DIR) if f.endswith(('.sql', '.dump'))]
            if not backup_files:
                return None
            return max(backup_files, key=lambda f: os.path.getmtime(os.path.join(settings.BACKUP_DIR, f)))
        except FileNotFoundError:
            return None
