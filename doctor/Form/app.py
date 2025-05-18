from flask import Flask, render_template, request, redirect, url_for, flash
import psycopg2
from psycopg2 import pool
import psycopg2.extras

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for flashing messages

# PostgreSQL connection details
db_uri = "postgres://avnadmin:AVNS_9fdLBoQqLpSjGyEZpe0@pg-1c91ea66-vimalrajses-df6c.i.aivencloud.com:27598/defaultdb?sslmode=require"

# Create a connection pool
connection_pool = psycopg2.pool.SimpleConnectionPool(1, 10, db_uri)


def get_connection():
    return connection_pool.getconn()


def return_connection(conn):
    connection_pool.putconn(conn)


@app.route('/')
def index():
    return render_template('form.html')


@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        room_no = request.form['room_no']
        patient_id = request.form['patient_id']
        patient_name = request.form['patient_name']
        physician_name = request.form['physician_name']

        conn = get_connection()
        try:
            cursor = conn.cursor()

            # Insert data - using both attending_physician and physician_name fields
            cursor.execute("""
                INSERT INTO table1 (room_no, patient_id, patient_name, attending_physician, physician_name) 
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (room_no) 
                DO UPDATE SET 
                    patient_id = EXCLUDED.patient_id,
                    patient_name = EXCLUDED.patient_name,
                    attending_physician = EXCLUDED.attending_physician,
                    physician_name = EXCLUDED.physician_name
            """, (room_no, patient_id, patient_name, physician_name, physician_name))

            conn.commit()
            flash('Patient information submitted successfully!')
        except Exception as e:
            conn.rollback()
            flash(f'Error: {str(e)}')
        finally:
            return_connection(conn)

        return redirect(url_for('view_patients'))


@app.route('/patients')
def view_patients():
    # Fetch patients from the database
    patients = []
    conn = get_connection()
    
    try:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        # Select all relevant fields from the table
        cursor.execute("""
            SELECT room_no, patient_id, patient_name, attending_physician
            FROM table1
        """)

        for row in cursor.fetchall():
            patients.append({
                'room_no': row['room_no'],
                'patient_id': row['patient_id'],
                'patient_name': row['patient_name'],
                'physician_name': row['attending_physician']  # Map attending_physician to physician_name for UI
            })
    except Exception as e:
        flash(f'Error: {str(e)}')
    finally:
        return_connection(conn)

    return render_template('patients.html', patients=patients)


if __name__ == '__main__':
    app.run(debug=True)
