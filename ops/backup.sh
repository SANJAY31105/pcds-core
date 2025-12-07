#!/bin/bash
# PCDS Database Backup Script
# Schedule this with crontab: 0 2 * * * /opt/pcds-enterprise/ops/backup.sh

# Configuration
BACKUP_DIR="/opt/pcds-enterprise/backups"
DB_CONTAINER="pcds-postgres"
DB_USER="pcds_admin"
DB_NAME="pcds_production"
DATE=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/opt/pcds-enterprise/logs/backup.log"

# Create backup directory if it doesn't exist
mkdir -p $BACKUP_DIR
mkdir -p /opt/pcds-enterprise/logs

echo "[$(date)] Starting backup job..." >> $LOG_FILE

# Check if container is running
if [ "$(docker ps -q -f name=$DB_CONTAINER)" ]; then
    # Create backup
    filename="$BACKUP_DIR/pcds_backup_$DATE.sql"
    
    if docker exec $DB_CONTAINER pg_dump -U $DB_USER $DB_NAME > "$filename"; then
        echo "[$(date)] Backup successful: $filename" >> $LOG_FILE
        
        # Compress backup
        gzip "$filename"
        echo "[$(date)] Compressed: $filename.gz" >> $LOG_FILE
        
        # Remove backups older than 30 days
        find $BACKUP_DIR -name "pcds_backup_*.sql.gz" -mtime +30 -delete
        echo "[$(date)] Cleanup of old backups complete" >> $LOG_FILE
    else
        echo "[$(date)] ERROR: Backup failed!" >> $LOG_FILE
        # Potential: send email alert here
    fi
else
    echo "[$(date)] ERROR: Container $DB_CONTAINER is not running!" >> $LOG_FILE
fi

echo "[$(date)] Backup job finished" >> $LOG_FILE
