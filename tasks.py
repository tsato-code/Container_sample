from invoke import task


@task
def stop(ctx):
    cmd = f"docker stop {ctx.container_name}"
    print(cmd)
    ctx.run(cmd)


@task(stop)
def remove(ctx):
    cmd = f"docker rm {ctx.container_name}"
    print(cmd)
    ctx.run(cmd)


@task
def run(ctx):
    cmd = f"docker run -t -d --name {ctx.container_name} -v $PWD:/tmp/working gcr.io/kaggle-images/python:v73"
    print(cmd)
    ctx.run(cmd)


@task
def prepare(ctx):
    cmd = f"docker exec -i -w /tmp/working {ctx.container_name} python project/source/prepare.py"
    print(cmd)
    ctx.run(cmd)


@task
def train(ctx):
    cmd = f"docker exec -i -w /tmp/working {ctx.container_name} python project/source/train.py"
    print(cmd)
    ctx.run(cmd)


@task
def evaluate(ctx):
    cmd = f"docker exec -i -w /tmp/working {ctx.container_name} python project/source/evaluate.py"
    print(cmd)
    ctx.run(cmd)

