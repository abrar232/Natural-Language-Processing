from locust import HttpUser, task, between

class NERServiceUser(HttpUser):
    wait_time = between(1, 5)
    host = "http://127.0.0.1:8000"

    @task
    def predict(self):
        payload = {"tokens": ["\"bob\", \"ross\", \"was\", \"an\", \"artist\", \".\""]}
        headers = {'Content-Type': 'application/json'}
        response = self.client.post("/predict/", json=payload, headers=headers)
        if response.status_code == 200:
            print(f"Success: {response.json()}")
        else:
            print(f"Failed: {response.status_code} {response.text}")

if __name__ == "__main__":
    import locust
    locust.run(["-f", __file__, "--host=http://127.0.0.1:8000"])


